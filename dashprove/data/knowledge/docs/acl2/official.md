───┬────────────────────────────────────────────────────────────────────────────────────────────────
[AC│# ACL2 Version 8.6                                                                              
L2 │                                                                                                
log│ACL2 is a logic and programming language in which you can model computer systems, together with 
o] │a tool to help you prove properties of those models. "ACL2" denotes "A Computational Logic for  
   │Applicative Common Lisp".                                                                       
   │                                                                                                
   │ACL2 is part of the Boyer-Moore family of provers, for which its authors have received the 2005 
   │[ACM Software System Award][1].                                                                 
   │                                                                                                
───┴────────────────────────────────────────────────────────────────────────────────────────────────

──────────────┬───────────────────────────────────────────┬──────────────┬──────────────────────────
[[door        │[Start Here][3] (including                 │[[teacher     │[ACL2 Workshops, UT       
icon]][2]     │[Applications][4], [Talks][5], [Tours][6], │icon]][8]     │Seminar, and Course       
              │and [Tutorials/Demos][7])                  │              │Materials][9]             
──────────────┼───────────────────────────────────────────┼──────────────┼──────────────────────────
[ [papers     │[ Publications about ACL2 and Its          │[[info bubble │[The User's Manuals][13]  
icon]][10]    │Applications][11]                          │icon]][12]    │and [Hyper-Card][14]      
──────────────┼───────────────────────────────────────────┼──────────────┼──────────────────────────
[[hammer and  │[Community Books: Lemma Libraries and      │[[mailbox     │[Mailing Lists][18]       
pliers        │Utilities][16]                             │icon]][17]    │                          
icon]][15]    │                                           │              │                          
──────────────┼───────────────────────────────────────────┼──────────────┼──────────────────────────
[ [NEW!       │[ Recent changes to this page][20]         │[[FTP         │[Obtaining, Installing,   
icon]][19]    │                                           │icon]][21]    │and License][22]          
──────────────┼───────────────────────────────────────────┼──────────────┼──────────────────────────
[[paper note  │[Differences from Version 8.5][24][ [tiny  │[ [filing     │[ Other Releases][27]     
icon]][23]    │warning icon]][25]                         │cabinet       │                          
              │                                           │icon]][26]    │                          
──────────────┴───────────────────────────────────────────┴──────────────┴──────────────────────────

[Matt Kaufmann][28] and [J Strother Moore][29]
[University of Texas at Austin][30]
October 10, 2024 (updated publications links June 1, 2025)


Welcome to the ACL2 home page! We highlight a few aspects of ACL2:

* Libraries (Books).
  Libraries of *books* (files containing definitions and theorems) extend the code that we have
  written. In particular, the distribution tarball includes the *community books*, which are
  contributed and maintained by the members of the ACL2 community.
* Documentation.
  There is an extensive user's manual for the ACL2 system, and an even more comprehensive manual
  that documents not only the ACL2 system but also many community books. See [below][31] to learn
  more.
* License and Copyright.
  ACL2 is freely available under the terms of the [LICENSE][32] file distributed with ACL2.
  [License, copyright, and authorship][33] information is available from the ACL2
  [documentation][34].
* Extensions.
  The ACL2 distribution includes the following extensions, which were developed by the individuals
  shown. NOTE:Not included in this list is what was formerly known as "ACL2(h)", because it is now
  the default build of ACL2: that is, ACL2 builds are now [hons-enabled][35]. Thanks to Bob Boyer,
  Warren A. Hunt, Jr., Jared Davis, and Sol Swords for their contributions; see the
  [acknowledgments][36].
  
  * [ACL2(r)][37]
    Support for the real numbers by way of non-standard analysis
    Ruben Gamboa
  * [ACL2(p)][38]
    Support for parallel evaluation
    David L. Rager
  Another extension of ACL2 is the Eclipse-based [ACL2 Sedan][39] (ACL2s). Unlike the systems above,
  ACL2s is distributed and maintained by Pete Manolios and his research group. ACL2s comes with a
  standard executable ACL2 image for Windows, but it also comes with pre-certified community books
  and an extension of ACL2 with additional features, including extra automation for termination
  proofs as well as counterexample generation.

We gratefully acknowledge substantial support from the sources listed in the [ACL2 acknowledgments
page][40].

## The User's Manuals

The *ACL2 User's Manual* is a vast hypertext document. If you are a newcomer to ACL2, we do *not*
recommend that you wander off into the full documentation. Instead start with the [START-HERE][41]
documentation topic. Experienced users tend mostly to use the manual as a reference manual, to look
up concepts mentioned in error messages or vaguely remembered from their past experiences with ACL2.

The *ACL2+Books Manual* includes not only the ACL2 User's Manual, but also documents many of the
[community books][42] (libraries). This manual, which is written by many authors, is thus more
extensive than the ACL2 system, and is thus potentially more useful. With the exception of the first
bulleted link below, links to the manual on this page will all take you to the ACL2+Books Manual.

The following links take you to these two manuals. The manuals can however be read not only in a Web
browser, but in the [ACL2-Doc Emacs browser][43] or by using the ACL2 `[:DOC][44]` command at the
terminal; see the documentation topic, `[DOCUMENTATION][45]`.

* [ACL2+Books Manual][46] (Version 8.6)
* [ACL2+Books Manual][47] (for [GitHub][48] distributions)
* [ACL2 User's Manual][49] (Version 8.6)

Once you have installed ACL2, you can browse the ACL2 User's Manual locally by viewing a copy of
this home page under your ACL2 sources directory at `doc/home-page.html` and following the last link
shown above; but first you will need to run the following command in your ACL2 sources directory.

    make DOC ACL2=<path_to_your_ACL2>

Better yet, you can build the ACL2+Books Manual locally, as follows, though this will likely take
longer (perhaps a half hour or more, depending on which books you have already certified).

    cd acl2-sources/books
    # The following uses ccl by default; sbcl is also acceptable.
    make manual ACL2=<path_to_your_ACL2>
The resulting ACL2+Books Manual may be accessed by pointing your browser to the file
`books/doc/manual/index.html` under your ACL2 sources directory.




## Community Books: Lemma Libraries and Utilities, and How to Contribute

A companion to ACL2 is the library of *community books*, which have been developed by many users
over the years. These books contain definitions and theorems that you might find useful in your
models and proofs. In addition, some books contain ACL2 tools built by users to help with reasoning,
programming, interfaces, debugging, and testing; see [ the documentation][50]. Some relevant papers
may be found by following links in the pages on [ Books and Papers about ACL2 and Its
Applications][51] and the [ACL2 Workshops Series][52]. The [installation instructions][53] explain
how to download and install the community books.

We strongly encourage users to submit additional books and to improve existing books. If you have
interest in contributing, there is a [documentation topic to get you started][54]. You can also
visit the `[ACL2 System and Books][55]` project page on github (just move past the big list of files
to find descriptive text). Project members are welcome to edit community books. In particular, the
community book `books/system/doc/acl2-doc.lisp` contains the ACL2 system documentation, and project
members are welcome to improve it.

(Prior to ACL2 Version 7.0 (January, 2015) books were [distributed through a different
mechanism][56].)


## Searching documentation

The web views of [The ACL2 User's Manual][57] and [ACL2+Books Manual][58] allow you to search the
short strings of the documentation (which are typically summaries of a line or so). To search the
full content for a string or regular expression, you may use the Emacs-based [ACL2-Doc browser][59].


[[Valid HTML 4.01 Transitional]][60]

[1]: http://awards.acm.org/software_system/
[2]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____The_02T
ours
[3]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____START-H
ERE
[4]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____INTERES
TING-APPLICATIONS
[5]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____TALKS
[6]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____The_02T
ours
[7]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____ACL2-TU
TORIAL
[8]: http://www.cs.utexas.edu/users/moore/acl2/workshops.html
[9]: http://www.cs.utexas.edu/users/moore/acl2/workshops.html
[10]: http://www.cs.utexas.edu/users/moore/acl2/manuals/current/manual/index.html?topic=ACL2____PUBL
ICATIONS
[11]: http://www.cs.utexas.edu/users/moore/acl2/manuals/current/manual/index.html?topic=ACL2____PUBL
ICATIONS
[12]: #User's-Manual
[13]: #User's-Manual
[14]: http://www.cs.utexas.edu/users/moore/publications/hyper-card.html
[15]: #Tools
[16]: #Tools
[17]: HTML/installation/misc.html#Addresses
[18]: HTML/installation/misc.html#Addresses
[19]: HTML/new.html
[20]: HTML/new.html
[21]: HTML/installation/installation.html
[22]: HTML/installation/installation.html
[23]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____NOTE-8
-6
[24]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____NOTE-8
-6
[25]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____A_02Ti
ny_02Warning_02Sign
[26]: HTML/other-releases.html
[27]: HTML/other-releases.html
[28]: mailto:kaufmann@cs.utexas.edu
[29]: mailto:moore@cs.utexas.edu
[30]: http://www.utexas.edu
[31]: #User's-Manual
[32]: HTML/LICENSE
[33]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____COPYRI
GHT
[34]: #documentation
[35]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____HONS-E
NABLED
[36]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____ACKNOW
LEDGMENTS
[37]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=COMMON-LISP___
_REAL
[38]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____PARALL
ELISM
[39]: http://acl2s.ccs.neu.edu/acl2s/doc/
[40]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____ACKNOW
LEDGMENTS
[41]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____START-
HERE
[42]: #Tools
[43]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____ACL2-D
OC
[44]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____DOC
[45]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=COMMON-LISP___
_DOCUMENTATION
[46]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/
[47]: http://www.cs.utexas.edu/users/moore/acl2/manuals/latest/
[48]: HTML/installation/obtaining-and-installing.html#GitHub
[49]: manual/index.html?topic=ACL2____ACL2
[50]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html
[51]: http://www.cs.utexas.edu/users/moore/acl2/manuals/current/manual/index.html?topic=ACL2____PUBL
ICATIONS
[52]: http://www.cs.utexas.edu/users/moore/acl2/workshops.html
[53]: HTML/installation/installation.html
[54]: http://www.cs.utexas.edu/users/moore/acl2/v8-6/combined-manual/index.html?topic=ACL2____GIT-QU
ICK-START
[55]: https://github.com/acl2/acl2
[56]: http://acl2.org/books-pre-7.0/
[57]: #User's-Manual
[58]: http://www.cs.utexas.edu/users/moore/acl2/current/combined-manual/
[59]: manual/index.html?topic=ACL2____ACL2-DOC
[60]: https://validator.w3.org/check?uri=referer
