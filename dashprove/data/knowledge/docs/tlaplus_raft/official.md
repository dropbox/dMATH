# The Raft Consensus Algorithm [Raft logo]

# Quick Links

* [Raft paper][1]
* [raft-dev mailing list][2]
* [Raft implementations][3]

## What is Raft?

Raft is a consensus algorithm that is designed to be easy to understand. It's equivalent to Paxos in
fault-tolerance and performance. The difference is that it's decomposed into relatively independent
subproblems, and it cleanly addresses all major pieces needed for practical systems. We hope Raft
will make consensus available to a wider audience, and that this wider audience will be able to
develop a variety of higher quality consensus-based systems than are available today.

## Hold on—what is consensus?

Consensus is a fundamental problem in fault-tolerant distributed systems. Consensus involves
multiple servers agreeing on values. Once they reach a decision on a value, that decision is final.
Typical consensus algorithms make progress when any majority of their servers is available; for
example, a cluster of 5 servers can continue to operate even if 2 servers fail. If more servers
fail, they stop making progress (but will never return an incorrect result).

Consensus typically arises in the context of replicated state machines, a general approach to
building fault-tolerant systems. Each server has a state machine and a log. The state machine is the
component that we want to make fault-tolerant, such as a hash table. It will appear to clients that
they are interacting with a single, reliable state machine, even if a minority of the servers in the
cluster fail. Each state machine takes as input commands from its log. In our hash table example,
the log would include commands like *set x to 3*. A consensus algorithm is used to agree on the
commands in the servers' logs. The consensus algorithm must ensure that if any state machine applies
*set x to 3* as the *n*^{th} command, no other state machine will ever apply a different *n*^{th}
command. As a result, each state machine processes the same series of commands and thus produces the
same series of results and arrives at the same series of states.

## Raft Visualization

Here's a Raft cluster running in your browser. You can interact with it to see Raft in action. Five
servers are shown on the left, and their logs are shown on the right. We hope to create a screencast
soon to explain what's going on. This visualization ([RaftScope][4]) is still pretty rough around
the edges; pull requests would be very welcome.

[The Secret Lives of Data][5] is a different visualization of Raft. It's more guided and less
interactive, so it may be a gentler starting point.

## Publications

This is "the Raft paper", which describes Raft in detail: [In Search of an Understandable Consensus
Algorithm (Extended Version)][6] by [Diego Ongaro][7] and [John Ousterhout][8]. A slightly shorter
version of this paper received a Best Paper Award at the [2014 USENIX Annual Technical
Conference][9].

Diego Ongaro's [Ph.D. dissertation][10] expands on the content of the paper in much more detail, and
it includes a simpler cluster membership change algorithm. The dissertation also includes a formal
specification of Raft written in TLA+; a slightly updated version of that specification is
[here][11].

More Raft-related papers:

* Doug Woos, James R. Wilcox, Steve Anton, Zachary Tatlock, Michael D. Ernst, and Thomas Anderson.
  [Planning for Change in a Formal Verification of the Raft Consensus Protocol][12].
  Certified Programs and Proofs (CPP), January 2016.
* James R. Wilcox, Doug Woos, Pavel Panchekha, Zachary Tatlock, Xi Wang, Michael D. Ernst, and
  Thomas Anderson.
  [Verdi: A Framework for Implementing and Verifying Distributed Systems][13].
  Programming Language Design and Implementation (PLDI), June 2015.
* Hugues Evrard and Frédéric Lang.
  [Automatic Distributed Code Generation from Formal Models of Asynchronous Concurrent
  Processes][14].
  Parallel, Distributed, and Network-Based Processing (PDP), March 2015.
* [Heidi Howard][15], Malte Schwarzkopf, Anil Madhavapeddy, and Jon Crowcroft.
  [Raft Refloated: Do We Have Consensus?][16].
  SIGOPS Operating Systems Review, January 2015.
* Heidi Howard.
  [ARC: Analysis of Raft Consensus][17].
  University of Cambridge, Computer Laboratory, UCAM-CL-TR-857, July 2014.

## Talks

These talks serve as good introductions to Raft:

* Talk on Raft at [CS@Illinois Distinguished Lecture Series][18] by [John Ousterhout][19], August
  2016: [preview of the video]
  
  ──────┬───────────────────────────────────────────
  Video │[YouTube][20]                              
  ──────┼───────────────────────────────────────────
  Slides│[PDF][21] with [RaftScope                  
        │visualization][22]                         
  ──────┴───────────────────────────────────────────
* Talk on Raft and its TLA+ spec as part of [Dr. TLA+ Series][23] by [Jin Li][24], July 2016:
  [preview of the video]
  
  ──────┬─────────────────
  Video │[YouTube][25]    
  ──────┼─────────────────
  Slides│[SlideShare][26] 
  ──────┴─────────────────
* Talk on Raft at [Build Stuff 2015][27] by [Diego Ongaro][28], November 2015: [preview of the
  video]
  
  ──────┬─────────────────────────────────────────────────────
  Video │[InfoQ][29]                                          
  ──────┼─────────────────────────────────────────────────────
  Slides│[HTML][30] [PDF][31] with [RaftScope                 
        │visualization][32]                                   
  ──────┴─────────────────────────────────────────────────────
* Talks on Rust, Raft, and distributed systems at [Rust Bay Area Meetup][33] by [Yvonne Coady][34],
  [Diego Ongaro][35], [Andrew Hobden][36], [Dan Burkert][37], and [Alex Newman][38], August 2015:
  [preview of the video]
  
  ──────┬─────────────────────────────────────────────────
  Video │[Air Mozilla][39]                                
  ──────┼─────────────────────────────────────────────────
  Slides│Diego: [PDF][40] with [RaftScope                 
        │visualization][41]                               
  ──────┴─────────────────────────────────────────────────
* Talk on Raft at [CoreOS Fest 2015][42] by [Diego Ongaro][43], May 2015: [preview of the video]
  
  ──────┬───────────────────────────────────────────
  Video │[YouTube][44]                              
  ──────┼───────────────────────────────────────────
  Slides│[PDF][45] with [RaftScope                  
        │visualization][46]                         
  ──────┴───────────────────────────────────────────
* Talk on Raft at [Sourcegraph meetup][47] by [Diego Ongaro][48], April 2015: [preview of the video]
  
  ──────┬───────────────────────────────────────────
  Video │[YouTube][49]                              
  ──────┼───────────────────────────────────────────
  Slides│[PDF][50] with [RaftScope                  
        │visualization][51]                         
  ──────┴───────────────────────────────────────────
* Talk on Raft at LinkedIn by [Diego Ongaro][52], September 2014: [preview of the video]
  
  ──────┬─────────────────────────────────────────────────────
  Video │[YouTube][53]                                        
  ──────┼─────────────────────────────────────────────────────
  Slides│[PDF][54] [PPTX][55] with [RaftScope                 
        │visualization][56]                                   
  ──────┴─────────────────────────────────────────────────────
* Talk on Raft at [USI 2014][57] and [/dev/summer 2014][58] by [Arnaud Bailly][59], July 2014:
  [preview of the video]
  
  ──────┬───────────────────────────
  Video │[YouTube][60] (French)     
  ──────┼───────────────────────────
  Slides│[Speaker Deck][61]         
        │(English)                  
  ──────┴───────────────────────────
* Talk on Raft at [2014 USENIX Annual Technical Conference][62] by [Diego Ongaro][63], June 2014:
  [preview of the video]
  
  ──────┬────────────────────────────
  Video │[USENIX][64]                
  ──────┼────────────────────────────
  Slides│[RaftScope                  
        │visualization][65]          
  ──────┴────────────────────────────
* Talk on Raft at [CraftConf 2014][66] by [Diego Ongaro][67], April 2014: [preview of the video]
  
  ──────┬────────────────────
  Video │[Ustream][68]       
  ──────┼────────────────────
  Slides│[PDF][69] [PPTX][70]
  ──────┴────────────────────
* Talk on Raft at [Rubyconf 2013][71] by [Patrick Van Stee][72], November 2013: [preview of the
  video]
  
  ──────┬───────────────────
  Video │[YouTube][73]      
  ──────┼───────────────────
  Slides│[Speaker Deck][74] 
  ──────┴───────────────────
* Talk on Raft at [RICON West 2013][75] by [Diego Ongaro][76], October 2013: [preview of the video]
  
  ──────┬────────────────────
  Video │[YouTube][77]       
  ──────┼────────────────────
  Slides│[PDF][78] [PPTX][79]
  ──────┴────────────────────
* Talk on Raft at [Strange Loop 2013][80] by [Ben Johnson][81], September 2013: [preview of the
  video]
  
  ──────┬─────────────────
  Video │[InfoQ][82]      
  ──────┼─────────────────
  Slides│[Speaker         
        │Deck][83]        
  ──────┴─────────────────
* Talk on Raft and [Rafter][84] at the [Erlang NYC Meetup][85] by [Tom Santero][86] and [Andrew
  Stone][87], August 2013: [preview of the video]
  
  ──────┬─────────────────
  Video │[Vimeo][88]      
  ──────┼─────────────────
  Slides│[Speaker         
        │Deck][89]        
  ──────┴─────────────────
* Talk on Raft (venue unknown) by [Patrick Van Stee][90], July 2013: [preview of the video]
  
  ──────┬─────────────────
  Slides│[Speaker         
        │Deck][91]        
  ──────┴─────────────────
* Lecture for the [Raft User Study][92] by [John Ousterhout][93], March 2013: [preview of the video]
  
  ──────────────────┬───────────────────────
  Video (screencast)│[YouTube][94] [MP4][95]
  ──────────────────┼───────────────────────
  Slides            │[PDF][96] [PPTX][97]   
  ──────────────────┴───────────────────────

## Courses teaching Raft

This is a list of courses that include lectures or programming assignments on Raft. This might be
useful for other instructors and for online learners looking for materials. If you know of
additional courses, please submit a [pull request][98] or an issue to update it.

* [The University of Hong Kong][99], [COMP3358: Distributed and Parallel Computing][100], [Heming
  Cui][101]. Includes lecture on Raft. (Spring 2024)
* [University of Virginia][102], [CS4740: Cloud Computing][103], [Chang Lou][104]. Includes lecture
  on Raft and programming assignment (Go). (Spring 2024 ...)
* [University of California, San Deigo][105], [CSE224: Graduate Networked Systems][106], [George
  Porter][107]. Includes lecture on Raft and programming assignment (Go) to develop a fault-tolerant
  Dropbox clone. (Winter 2023/2022, Fall 2020/2019/2018 ...)
* [Technical University of Munich][108], [IN2259: Distributed Systems][109], [Pramod Bhatotia][110]
  and [Martin Kleppmann][111]. Includes lecture on Raft. (Winter 2022/2023, ...)
* [University of Illinois Urbana-Champaign][112], [CS425: Distributed Systems][113], [Indranil
  Gupta][114], [Nikita Borisov][115], and [Radhika Mittal][116]. Includes Raft programming
  assignment in Go. (Spring 2021, ...)
* [The Chinese University of Hong Kong][117], [CSCI4160: Distributed and Parallel Computing][118]
  (Private), [Eric Lo][119]. Includes lecture on Paxos and Raft as well as [Raft programming
  assignment][120] in Java/Go (Fall 2019, Spring 2021, ...)
* [Indian Institute of Technology Delhi][121], [COL 819: Distributed Systems][122], [Smruti R.
  Sarangi][123]. Includes lecture on Raft ([video][124]) ([PPTX][125]) (Winter 2020, ...)
* [Carnegie Mellon University][126], [CS 440: Distributed Systems][127], [Yuvraj Agarwal][128],
  [Dave Andersen][129]. Includes Raft programming assignment in Go. (Fall 2019, ...)
* [Northeastern University][130], [CS 3700: Networks and Distributed Systems][131], [Christo
  Wilson][132], [Long Lu][133]. Includes an assignment to build a replicated key-value store based
  on the Raft protocol. (Fall 2018, ...)
* [Institute Technology Bandung][134], IF3230. Raft [assignments][135] with testcases for showcasing
  edge cases and scoring (2018)
* [Rose-Hulman Institute of Technology][136], [CS 403: Programming Language Paradigms][137],
  [Buffalo Hewner][138]. Includes Raft programming assignment in Erlang ([assignments][139]).
  (Winter 2017, ...)
* [Princeton University][140], [COS-418: Distributed Systems][141], [Mike Freedman][142] and [Kyle
  Jamieson][143]. Includes [lecture on Raft][144] ([PPTX][145]) and [programming assignments][146]
  to build a Raft-based key-value store. (Fall 2016, ...)
* [University of Washington][147], [CSE 452: Distributed Systems][148], [Tom Anderson][149].
  Includes [lecture on Raft][150], though they call it Paxos. (Winter 2016, ...)
* [University of Colorado, Boulder][151], [CSCI 5673: Distributed Systems][152], [Shivakant
  Mishra][153]. Includes assignment to download a Raft implementation and build a fault-tolerant
  data structure with it. (Fall 2015, ...)
* [University of Utah][154], [CS 6963: Distributed Systems][155], [Ryan Stutsman][156]
  ([@rstutsman][157]). Will include something about Raft (TBD). (Fall 2015, ...)
* [San Jose State University][158], [CMPE 275 Enterprise Application Development][159], [John
  Gash][160]. Includes project to make a [distributed filesystem][161] using Raft. (Spring 2015,
  ...)
* [Brown][162], [CS 138: Distributed Computer Systems][163], [Tom Doeppner][164], [Rodrigo
  Fonseca][165] ([@rodrigo_fonseca][166]). Includes Raft programming assignment in Go. (Spring 2015,
  ...)
* [MIT][167], [6.824: Distributed Systems][168], [Robert Morris][169]. Includes lecture on Raft
  ([lecture notes][170]). See [Jon Gjengset][171]'s posts for [instructors][172] and
  [students][173]. (Spring 2015, ...)
* [University of San Francisco][174], [CS 636: Graduate Operating Systems][175], [Greg Benson][176]
  ([@gregorydbenson][177]). Includes lecture on Raft. (Spring 2015, ...)
* [Harvard][178], [CS 261: Research Topics in Operating Systems][179], [Margo Seltzer][180].
  Includes lecture on Raft ([lecture notes][181]). (Fall 2014, ...)
* [University of Houston][182], [COSC 6360: Operating Systems][183], [Jehan-François Pâris][184]
  ([@jehanfrancois][185]). Includes lecture on Raft ([PPT][186]). (Fall 2014, ...)
* [Stanford][187], [CS 244b: Distributed Systems][188], [Dawson Engler][189], [David Mazières][190]
  ([@dmazieres][191]). Included guest lecture on Raft by Diego Ongaro. Several students chose to
  work on Raft-based [final projects][192]. (Fall 2014)
* [NUST-SEECS][193], CS 332: Distributed Computing, Tahir Azim ([@TahirAzim][194]). Includes lecture
  on Raft based on user study materials ([tweet][195]). (Fall 2014, ...)
* [Duke][196], [CPS 512: Distributed Systems][197], [Bruce Maggs][198]. Includes guest lecture on
  Raft ([PPTX][199]) by [Landon Cox][200] ([@lpcox][201]). (Spring 2014, Spring 2015, ...)
* [IIT Bombay][202], [CS 733: Cloud Computing][203], [Sriram Srinivasan][204]. Includes Raft
  programming assignment in Go ([assignments][205]). (Spring 2014, Spring 2015, ...)
* [Polytechnic University of Milan][206], [Distributed Systems][207], [Gianpaolo Cugola][208].

## Where can I ask questions?

The best place to ask questions about Raft and its implementations is the [raft-dev Google
group][209]. Some of the implementations also have their own mailing lists; check their READMEs.

## Where can I get Raft?

There are many implementations of Raft available in various stages of development. This table lists
the implementations we know about with source code available. The most popular and/or recently
updated implementations are towards the top. This information will inevitably get out of date;
please submit a [pull request][210] or an issue to update it.

────┬───┬────────────┬──────┬─────┬─────────────────────────┬──────────┬───────────────┬────────────
Star│Nam│Primary     │Langua│Licen│Leader Election + Log    │Persistenc│Membership     │Log         
s   │e  │Authors     │ge    │se   │Replication?             │e?        │Changes?       │Compaction? 
────┴───┴────────────┴──────┴─────┴─────────────────────────┴──────────┴───────────────┴────────────

Published with [GitHub Pages][211]. [View on GitHub][212].
This work is licensed under a [Creative Commons Attribution 3.0 Unported License][213].

[1]: raft.pdf
[2]: https://groups.google.com/forum/#!forum/raft-dev
[3]: #implementations
[4]: https://github.com/ongardie/raftscope
[5]: http://thesecretlivesofdata.com/raft/
[6]: raft.pdf
[7]: https://twitter.com/ongardie
[8]: https://www.stanford.edu/~ouster/
[9]: https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro
[10]: https://github.com/ongardie/dissertation#readme
[11]: https://github.com/ongardie/raft.tla
[12]: https://dl.acm.org/doi/abs/10.1145/2854065.2854081
[13]: https://dl.acm.org/doi/pdf/10.1145/2737924.2737958
[14]: https://hal.inria.fr/hal-01086522
[15]: https://twitter.com/heidiann360
[16]: https://api.repository.cam.ac.uk/server/api/core/bitstreams/c9bcee5b-a1cb-4147-9281-1a05632f5a
a3/content
[17]: https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-857.html
[18]: https://cs.illinois.edu/news/distinguished-lecture-series-dr-john-ousterhout
[19]: https://www.stanford.edu/~ouster/
[20]: https://youtu.be/vYp4LYbnnW8
[21]: slides/uiuc2016.pdf
[22]: raftscope/index.html
[23]: https://github.com/tlaplus/DrTLAPlus
[24]: https://research.microsoft.com/en-us/um/people/jinl/
[25]: https://www.youtube.com/watch?v=6Kwx8zfGW0Y
[26]: https://www.slideshare.net/DrTlaplusSeries/dr-tla-series-raft-jin-li
[27]: https://www.buildstuff.events/
[28]: https://twitter.com/ongardie
[29]: https://www.infoq.com/presentations/raft-consensus-algorithm
[30]: https://ongardie.github.io/raft-talk-archive/2015/buildstuff/
[31]: slides/buildstuff2015.pdf
[32]: https://ongardie.github.io/raft-talk-archive/2015/buildstuff/raftscope-replay/
[33]: https://www.meetup.com/Rust-Bay-Area/events/219696985/
[34]: https://webhome.cs.uvic.ca/~ycoady/
[35]: https://twitter.com/ongardie
[36]: https://twitter.com/andrewhobden
[37]: https://github.com/danburkert
[38]: https://twitter.com/posix4e
[39]: https://air.mozilla.org/bay-area-rust-meetup-august-2015/
[40]: slides/rustdiego2015.pdf
[41]: raftscope-replay/index.html
[42]: https://coreos.com/fest/
[43]: https://twitter.com/ongardie
[44]: https://youtu.be/6bBggO6KN_k
[45]: slides/coreosfest2015.pdf
[46]: raftscope-replay/index.html
[47]: https://www.meetup.com/Sourcegraph-Hacker-Meetup/events/221199291/
[48]: https://twitter.com/ongardie
[49]: https://youtu.be/2dfSOFqOhOU
[50]: slides/sourcegraph2015.pdf
[51]: raftscope-replay/index.html
[52]: https://twitter.com/ongardie
[53]: https://youtu.be/LAqyTyNUYSY
[54]: slides/linkedin2014.pdf
[55]: slides/linkedin2014.pptx
[56]: raftscope-replay/index.html
[57]: https://www.usievents.com/en
[58]: http://devcycles.net/summer/sessions/index.php?session=3
[59]: https://twitter.com/abailly
[60]: https://www.youtube.com/watch?v=eRDq2Fr6grY
[61]: https://speakerdeck.com/abailly/the-raft-protocol-distributed-consensus-for-dummies
[62]: https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro
[63]: https://twitter.com/ongardie
[64]: https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro
[65]: raftscope-replay/index.html
[66]: https://craft-conf.com/2014/#speakers/DiegoOngaro
[67]: https://twitter.com/ongardie
[68]: https://www.ustream.tv/recorded/46672856
[69]: slides/craftconf2014.pdf
[70]: slides/craftconf2014.pptx
[71]: https://rubyconf.org/program#patrick-van-stee
[72]: https://twitter.com/vanstee
[73]: https://youtu.be/IsPxhZ2IsWw
[74]: https://speakerdeck.com/vanstee/raft-consensus-for-rubyists
[75]: http://ricon.io/west.html
[76]: https://twitter.com/ongardie
[77]: https://youtu.be/06cTPhi-3_8
[78]: slides/riconwest2013.pdf
[79]: slides/riconwest2013.pptx
[80]: https://thestrangeloop.com/sessions/raft-the-understandable-distributed-protocol
[81]: https://twitter.com/benbjohnson
[82]: https://www.infoq.com/presentations/raft
[83]: https://speakerdeck.com/benbjohnson/raft-the-understandable-distributed-consensus-protocol
[84]: https://github.com/andrewjstone/rafter
[85]: https://www.meetup.com/Erlang-NYC/events/131394712/
[86]: https://twitter.com/tsantero
[87]: https://twitter.com/andrew_j_stone
[88]: https://vimeo.com/71635670
[89]: https://speakerdeck.com/tsantero/consensus-raft-and-rafter
[90]: https://twitter.com/vanstee
[91]: https://speakerdeck.com/vanstee/consensus-an-introduction-to-raft
[92]: https://ongardie.net/static/raft/userstudy/
[93]: https://www.stanford.edu/~ouster/
[94]: https://youtu.be/YbZ3zDzDnrw
[95]: https://ongardie.net/static/raft/userstudy/raft.mp4
[96]: slides/raftuserstudy2013.pdf
[97]: slides/raftuserstudy2013.pptx
[98]: https://github.com/raft/raft.github.io
[99]: https://www.hku.hk/
[100]: https://www.cs.hku.hk/index.php/programmes/course-offered?infile=2019/comp3358.html
[101]: https://www.cs.hku.hk/people/academic-staff/heming
[102]: https://www.virginia.edu/
[103]: https://changlousys.github.io/CS4740/spring24/about/
[104]: https://changlousys.github.io/
[105]: https://ucsd.edu
[106]: https://canvas.ucsd.edu/courses/43955/assignments/syllabus
[107]: https://cseweb.ucsd.edu/~gmporter/
[108]: https://www.tum.de/
[109]: https://campus.tum.de/tumonline/ee/ui/ca2/app/desktop/#/slc.tm.cp/student/courses/950635146?l
ang=en
[110]: https://dse.in.tum.de/bhatotia/
[111]: https://martin.kleppmann.com/
[112]: https://illinois.edu
[113]: https://courses.grainger.illinois.edu/cs425/
[114]: http://indy.cs.illinois.edu
[115]: http://hatswitch.org/~nikita/
[116]: https://radhikam.web.illinois.edu
[117]: https://www.cse.cuhk.edu.hk
[118]: https://piazza.com/class/kicgvsku8ul6ro
[119]: https://appsrv.cse.cuhk.edu.hk/~ericlo/
[120]: https://github.com/eric-lo/csci4160-asgn1
[121]: https://www.cse.iitd.ac.in/
[122]: https://www.cse.iitd.ernet.in/~srsarangi/courses/2020/col_819_2020/index.html
[123]: https://www.cse.iitd.ac.in/~srsarangi/
[124]: https://youtu.be/oBlbpPDxs-M
[125]: https://www.cse.iitd.ernet.in/~srsarangi/courses/2020/col_819_2020/docs/raft.pptx
[126]: https://www.cs.cmu.edu/
[127]: https://www.synergylabs.org/courses/15-440/
[128]: https://www.synergylabs.org/yuvraj/
[129]: https://www.cs.cmu.edu/~dga/
[130]: https://khoury.northeastern.edu/
[131]: https://cbw.sh/3700/
[132]: https://cbw.sh/
[133]: https://khoury.northeastern.edu/people/long-lu/
[134]: https://itb.ac.id/
[135]: https://github.com/ramsicandra/IF3230-Raft-2018
[136]: https://www.rose-hulman.edu/
[137]: http://files.hewner.com/classes/csse403/
[138]: http://hewner.com
[139]: http://files.hewner.com/classes/csse403/HomeworkCode/ErlangRaft/
[140]: https://www.cs.princeton.edu/
[141]: https://www.cs.princeton.edu/courses/archive/fall16/cos418/
[142]: https://www.cs.princeton.edu/~mfreed/
[143]: https://www.cs.princeton.edu/~kylej/
[144]: https://www.cs.princeton.edu/courses/archive/fall16/cos418/docs/L8-consensus-2.pdf
[145]: https://www.cs.princeton.edu/courses/archive/fall16/cos418/docs/L8-consensus-2.pptx
[146]: https://www.cs.princeton.edu/courses/archive/fall16/cos418/assignments.html
[147]: https://www.cs.washington.edu/
[148]: https://courses.cs.washington.edu/courses/cse452/16wi/
[149]: https://www.cs.washington.edu/people/faculty/tom
[150]: https://courses.cs.washington.edu/courses/cse452/16wi/calendar/calendar.html
[151]: https://www.cs.colorado.edu/
[152]: https://www.cs.colorado.edu/~mishras/courses/csci5673/Fall15/
[153]: https://www.cs.colorado.edu/~mishras/
[154]: https://www.cs.utah.edu/
[155]: https://www.cs.utah.edu/~stutsman/cs6963/
[156]: https://www.cs.utah.edu/~stutsman/
[157]: https://twitter.com/rstutsman
[158]: http://info.sjsu.edu/web-dbgen/catalog/departments/CMPE.html
[159]: http://info.sjsu.edu/web-dbgen/catalog/courses/CMPE275.html
[160]: https://cmpe.sjsu.edu/profile/john-gash
[161]: https://github.com/deepmehtait/Distributed-file-system-server-with-RAFT
[162]: https://cs.brown.edu/
[163]: https://cs.brown.edu/courses/csci1380/
[164]: https://www.cs.brown.edu/~twd/
[165]: https://www.cs.brown.edu/~rfonseca/
[166]: https://twitter.com/rodrigo_fonseca
[167]: https://www.csail.mit.edu/
[168]: http://nil.csail.mit.edu/6.824/2015/index.html
[169]: https://pdos.csail.mit.edu/~rtm/
[170]: http://nil.csail.mit.edu/6.824/2015/notes/l-raft.txt
[171]: https://twitter.com/Jonhoo/
[172]: https://thesquareplanet.com/blog/instructors-guide-to-raft/
[173]: https://thesquareplanet.com/blog/students-guide-to-raft/
[174]: http://cs.usfca.edu/
[175]: http://cs636.cs.usfca.edu/home
[176]: http://benson.cs.usfca.edu/
[177]: https://twitter.com/gregorydbenson
[178]: https://www.eecs.harvard.edu/
[179]: https://www.eecs.harvard.edu/cs261/
[180]: https://www.eecs.harvard.edu/margo
[181]: https://www.eecs.harvard.edu/cs261/notes/ongara-2014.html
[182]: https://uh.edu/nsm/computer-science/
[183]: http://www2.cs.uh.edu/~paris/6360/resources.htm
[184]: http://www2.cs.uh.edu/~paris/
[185]: https://twitter.com/jehanfrancois
[186]: http://www2.cs.uh.edu/~paris/6360/PowerPoint/Raft.ppt
[187]: https://cs.stanford.edu/
[188]: https://www.scs.stanford.edu/14au-cs244b/
[189]: https://web.stanford.edu/~engler/
[190]: https://www.scs.stanford.edu/~dm/
[191]: https://twitter.com/dmazieres
[192]: https://www.scs.stanford.edu/14au-cs244b/labs/presentations.html
[193]: http://seecs.nust.edu.pk/
[194]: https://twitter.com/TahirAzim
[195]: https://twitter.com/TahirAzim/status/527363109678112768
[196]: https://www.cs.duke.edu/
[197]: http://db.cs.duke.edu/courses/compsci512/spring15/
[198]: https://www.cs.duke.edu/~bmm/
[199]: http://db.cs.duke.edu/courses/compsci512/spring15/lectures/raft-guest.pptx
[200]: https://www.cs.duke.edu/~lpcox/
[201]: https://twitter.com/lpcox
[202]: https://www.cse.iitb.ac.in/
[203]: https://www.cse.iitb.ac.in/page134?course=CS+733
[204]: https://github.com/sriram-srinivasan
[205]: https://github.com/dushyant89/CS-733
[206]: https://polimi.it/
[207]: https://www4.ceda.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_R
IGA_MANIFESTO=evento&aa=2020&k_cf=225&k_corso_la=481&k_indir=T2A&codDescr=090950&lang=IT&semestre=1&
idGruppo=4151&idRiga=253827
[208]: https://www4.ceda.polimi.it/manifesti/manifesti/controller/ricerche/RicercaPerDocentiPublic.d
o?evn_didattica=evento&k_doc=51387&aa=2020&lang=IT&jaf_currentWFID=main
[209]: https://groups.google.com/forum/#!forum/raft-dev
[210]: https://github.com/raft/raft.github.io
[211]: https://pages.github.com
[212]: https://github.com/raft/raft.github.io
[213]: https://creativecommons.org/licenses/by/3.0/deed.en_US
