────────────
[[banner]]  
────────────

─


## Verifying
## Multi-threaded
## Software
## with Spin


[[logo]]

─┬─┬────────────────────────────────────────────────────────────────────────────────────────────┬─┬─
 │ │Spin is a widely used open-source software verification tool. The tool can be used for the  │ │ 
 │ │formal verification of multi-threaded software applications. The tool was developed at Bell │ │ 
 │ │Labs in the Unix group of the Computing Sciences Research Center, starting in 1980, and has │ │ 
 │ │been available freely since 1991. Spin continues to evolve to keep pace with new            │ │ 
 │ │developments in the field. In April 2002 the tool was awarded the ACM [ System Software     │ │ 
 │ │Award][1]. [[read more] ][2]                                                                │ │ 
─┴─┴────────────────────────────────────────────────────────────────────────────────────────────┴─┴─

─┬────────────────────────┬─────────────────────────┬──────────────────────┬──────────────────────┬─
 │discover                │learn                    │use                   │community             │ 
 │                        │                         │                      │                      │ 
 │* [what is spin?][3]    │* [tutorials][7]         │* [installation][12]  │* [forum][16]         │ 
 │* [success stories][4]  │* [books][8]             │* [man pages][13]     │* [symposia][17]      │ 
 │* [examples][5]         │* [papers][9]            │* [options][14]       │* [support][18]       │ 
 │* [roots][6]            │* [model extraction][10] │* [releases][15]      │* [projects][19]      │ 
 │                        │* [exercises][11]        │                      │                      │ 
─┴────────────────────────┴─────────────────────────┴──────────────────────┴──────────────────────┴─

─┬────────────────────────────────────────────────────────────────────────────────────────────────┬─
 │*Open Source:* Starting with Version 6.4.5 from January 2016, the Spin sources are available    │ 
 │under the standard BSD 3-Clause open source license. Spin is now also part of the latest stable │ 
 │release of Debian Linux, and has made it into the 16.10+ distributions of Ubuntu. The current   │ 
 │Spin version is 6.5.1 (July 2020).                                                              │ 
─┴────────────────────────────────────────────────────────────────────────────────────────────────┴─

─┬────────────────────────────────────────────────────────────────────────────────────────────────┬─
 │*Symposia:* The [32nd International Symposium on Model Checking Software][20] will be held April│ 
 │15-16 2026 in Torino, Italy, co-located with [ETAPS-2026][21]. The Symposium is organized by    │ 
 │[Vincenzo Ciancia][22] and [Arnd Hartmanns][23].                                                │ 
─┴────────────────────────────────────────────────────────────────────────────────────────────────┴─

─┬────────────────────────────────────────────────────────────────────────────────────────────────┬─
 │*New Book on Static Code Analysis with Cobra:* Take a look at: [https://codescrub.com][24] where│ 
 │you can find information on a new, very efficient, approach to interactive static source code   │ 
 │analysis. The Cobra tool is fast enough to be used on very large code bases. A comprehensive new│ 
 │book describing how to become an expert user is available now on [amazon][25]. Cobra is open    │ 
 │source and available on github. Be one of the first reviewers!                                  │ 
─┴────────────────────────────────────────────────────────────────────────────────────────────────┴─

─┬────────────────────────────────────────────────────────────────────────────────────────────────┬─
 │*Courses:* A short [online course][26] in software verification and logic model checking is     │ 
 │available (password required). There are a total 15 short lectures covering the                 │ 
 │automata-theoretic verification method, the basic use of Spin, model extraction from C source   │ 
 │code, abstraction methods, and swarm verification techniques. You can see an overview via this  │ 
 │[link][27]. An excellent introduction to the basics of model checking.                          │ 
 │                                                                                                │ 
 │*In-Depth:* A full one semester college-level course is also available, complete with           │ 
 │transcripts of every lecture, quizzes, assignments, and exercises to test your understanding and│ 
 │practice new skills. Details can be found in this [syllabus][28].                               │ 
─┴────────────────────────────────────────────────────────────────────────────────────────────────┴─

─┬────────────────────────────────────────────────────────────────────────────────────────────────┬─
 │*Tau Tool:* A simple front-end tool for Spin, called *Tau* (short for *Tiny Automata*) can be   │ 
 │downloaded from: [https://spinroot.com/spin/tau_v1.tar.gz][29], and is distributed under        │ 
 │[LGPL][30], originally by Caltech, as a teaching tool for formal verification and finite        │ 
 │automata.                                                                                       │ 
─┴────────────────────────────────────────────────────────────────────────────────────────────────┴─

─┬────────────────────────────────────────────────────────────────────────────────────────────────┬─
 │    // a small example spin model                                                               │ 
 │    // Peterson's solution to the mutual exclusion problem (1981)                               │ 
 │                                                                                                │ 
 │    bool turn, flag[2];         // the shared variables, booleans                               │ 
 │    byte ncrit;                 // nr of procs in critical section                              │ 
 │                                                                                                │ 
 │    active [2] proctype user()  // two processes                                                │ 
 │    {                                                                                           │ 
 │        assert(_pid == 0 || _pid == 1);                                                         │ 
 │    again:                                                                                      │ 
 │        flag[_pid] = 1;                                                                         │ 
 │        turn = _pid;                                                                            │ 
 │        (flag[1 - _pid] == 0 || turn == 1 - _pid);                                              │ 
 │                                                                                                │ 
 │        ncrit++;                                                                                │ 
 │        assert(ncrit == 1);     // critical section                                             │ 
 │        ncrit--;                                                                                │ 
 │                                                                                                │ 
 │        flag[_pid] = 0;                                                                         │ 
 │        goto again                                                                              │ 
 │    }                                                                                           │ 
 │    // analysis:                                                                                │ 
 │    // $ spin -run peterson.pml                                                                 │ 
─┴────────────────────────────────────────────────────────────────────────────────────────────────┴─

─┬──────────────────────────────────────────┬──
 │[[revert to the old spin homepage]][31]   │  
─┴──────────────────────────────────────────┴──

[1]: https://awards.acm.org/software_system
[2]: https://spinroot.com/spin/what.html
[3]: https://spinroot.com/spin/what.html
[4]: https://spinroot.com/spin/success.html
[5]: https://www.imm.dtu.dk/~albl/promela.html
[6]: https://spinroot.com/spin/Doc/roots.html
[7]: https://spinroot.com/spin/Man/
[8]: https://spinroot.com/spin/books.html
[9]: https://spinroot.com/spin/theory.html
[10]: https://spinroot.com/modex/
[11]: https://spinroot.com/courses/summer
[12]: https://spinroot.com/spin/Man/README.html
[13]: https://spinroot.com/spin/Man/promela.html
[14]: https://spinroot.com/spin/Man/Spin.html
[15]: https://spinroot.com/spin/Doc/V6.Updates
[16]: https://spinroot.com/forum/
[17]: https://spinroot.com/spin/symposia/
[18]: https://spinroot.com/spin/old.html#C
[19]: https://spinroot.com/spin/projects.html
[20]: https://spin-web.github.io/SPIN2026/
[21]: https://etaps.org
[22]: https://vincenzoml.github.io/
[23]: https://arnd.hartmanns.name
[24]: https://codescrub.com
[25]: https://www.amazon.com/dp/B0FFVZC8F1?ref_=pe_93986420_774957520
[26]: https://spinroot.com/course
[27]: https://spinroot.com/course
[28]: syllabus.html
[29]: https://spinroot.com/spin/tau_v1.tar.gz
[30]: https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License
[31]: https://spinroot.com/spin/old.html
