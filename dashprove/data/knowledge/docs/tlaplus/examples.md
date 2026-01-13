# TLA^{+} Examples

[[Gitpod ready-to-code]][1] [[Validate Specs & Models]][2]

This is a repository of TLA^{+} specifications and models covering applications in a variety of
fields. It serves as:

* a comprehensive example library demonstrating how to specify an algorithm in TLA^{+}
* a diverse corpus facilitating development & testing of TLA^{+} language tools
* a collection of case studies in the application of formal specification in TLA^{+}

All TLA^{+} specs can be found in the [`specifications`][3] directory. To contribute a spec of your
own, see [`CONTRIBUTING.md`][4].

The table below lists all specs and indicates whether a spec is beginner-friendly, includes an
additional PlusCal variant `(✔)`, or uses PlusCal exclusively. Additionally, the table specifies
which verification tool—[TLC][5], [Apalache][6], or [TLAPS][7]—can be used to verify each
specification.

Space contraints limit the information displayed in the table; detailed spec metadata can be found
in the `manifest.json` files in each specification's directory. You can search these files for
examples exhibiting a number of features, either using the GitHub repository search or locally with
the command `ls specifications/*/manifest.json | xargs grep -l $keyword`, where `$keyword` can be a
value like:

* `pluscal`, `proof`, or `action composition` (the `\cdot` operator)
* Specs intended for trace generation (`generate`), `simulate`, or checked symbolically with
  Apalache (`symbolic`)
* Models failing in interesting ways, like `deadlock failure`, `safety failure`, `liveness failure`,
  or `assumption failure`

It is also helpful to consult model files using specific TLC features. For this, run `ls
specifications/*/*.cfg | xargs grep -l $keyword`, where `$keyword` can be a feature like `SYMMETRY`,
`VIEW`, `ALIAS`, `CONSTRAINT`, or `DEADLOCK`.

## Validated Examples Included Here

Here is a list of specs included in this repository which are validated by the CI, with links to the
relevant directory and flags for various features:

──────────────────────────────────────────┬─────────────────────────────┬────┬──────┬────┬─────┬────
Name                                      │Author(s)                    │Begi│TLAPS │Plus│TLC  │Apal
                                          │                             │nner│Proof │Cal │Model│ache
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Teaching Concurrency][8]                 │Leslie Lamport               │✔   │✔     │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Loop Invariance][9]                      │Leslie Lamport               │✔   │✔     │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Learn TLA⁺ Proofs][10]                   │Andrew Helwer                │✔   │✔     │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Boyer-Moore Majority Vote][11]           │Stephan Merz                 │✔   │✔     │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Proof x+x is Even][12]                   │Martin Riener                │✔   │✔     │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The N-Queens Puzzle][13]                 │Stephan Merz                 │✔   │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Dining Philosophers Problem][14]     │Jeff Hemphill                │✔   │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Car Talk Puzzle][15]                 │Leslie Lamport               │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Die Hard Problem][16]                │Leslie Lamport               │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Prisoners & Switches Puzzle][17]     │Leslie Lamport               │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Prisoners & Switch Puzzle][18]       │Florian Schanda              │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Specs from Specifying Systems][19]       │Leslie Lamport               │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Tower of Hanoi Puzzle][20]           │Markus Kuppe, Alexander      │✔   │      │    │✔    │    
                                          │Niederbühl                   │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Missionaries and Cannibals][21]          │Leslie Lamport               │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Stone Scale Puzzle][22]                  │Leslie Lamport               │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Coffee Can Bean Problem][23]         │Andrew Helwer                │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[EWD687a: Detecting Termination in        │Stephan Merz, Leslie Lamport,│✔   │      │(✔) │✔    │    
Distributed Computations][24]             │Markus Kuppe                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Moving Cat Puzzle][25]               │Florian Schanda              │✔   │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Boulangerie Algorithm][26]           │Leslie Lamport, Stephan Merz │    │✔     │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Misra Reachability Algorithm][27]        │Leslie Lamport               │    │✔     │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Byzantizing Paxos by Refinement][28]     │Leslie Lamport               │    │✔     │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Barrier Synchronization][29]             │Jarod Differdange            │    │✔     │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Peterson Lock Refinement With Auxiliary  │Jarod Differdange            │    │✔     │✔   │✔    │    
Variables][30]                            │                             │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[EWD840: Termination Detection in a       │Stephan Merz                 │    │✔     │    │✔    │    
Ring][31]                                 │                             │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[EWD998: Termination Detection in a Ring  │Stephan Merz, Markus Kuppe   │    │✔     │(✔) │✔    │    
with Asynchronous Message Delivery][32]   │                             │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Paxos Protocol][33]                  │Leslie Lamport               │    │(✔)   │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Asynchronous Reliable Broadcast][34]     │Thanh Hai Tran, Igor Konnov, │    │✔     │    │✔    │    
                                          │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Distributed Mutual Exclusion][35]        │Stephan Merz                 │    │✔     │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Two-Phase Handshaking][36]               │Leslie Lamport, Stephan Merz │    │✔     │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Paxos (How to Win a Turing Award)][37]   │Leslie Lamport               │    │(✔)   │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Finitizing Monotonic Systems][38]        │Andrew Helwer, Stephan Merz, │    │✔     │    │✔    │    
                                          │Markus Kuppe                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Dijkstra's Mutual Exclusion              │Leslie Lamport               │    │      │✔   │✔    │    
Algorithm][39]                            │                             │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Echo Algorithm][40]                  │Stephan Merz                 │    │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The TLC Safety Checking Algorithm][41]   │Markus Kuppe                 │    │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Transaction Commit Models][42]           │Leslie Lamport, Jim Gray,    │    │      │✔   │✔    │    
                                          │Murat Demirbas               │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Slush Protocol][43]                  │Andrew Helwer                │    │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Minimal Circular Substring][44]          │Andrew Helwer                │    │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Snapshot Key-Value Store][45]            │Andrew Helwer, Murat Demirbas│    │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Chang-Roberts Algorithm for Leader       │Stephan Merz                 │    │      │✔   │✔    │    
Election in a Ring][46]                   │                             │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[MultiPaxos in SMR-Style][47]             │Guanzhou Hu                  │    │      │✔   │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Einstein's Riddle][48]                   │Isaac DeFrain, Giuliano Losa │    │      │    │     │✔   
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Resource Allocator][49]                  │Stephan Merz                 │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Transitive Closure][50]                  │Stephan Merz                 │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Atomic Commitment Protocol][51]          │Stephan Merz                 │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[SWMR Shared Memory Disk Paxos][52]       │Leslie Lamport, Giuliano Losa│    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Span Tree Exercise][53]                  │Leslie Lamport               │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Knuth-Yao Method][54]                │Ron Pressler, Markus Kuppe   │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Huang's Algorithm][55]                   │Markus Kuppe                 │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[EWD 426: Token Stabilization][56]        │Murat Demirbas, Markus Kuppe │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Sliding Block Puzzle][57]                │Mariusz Ryndzionek           │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Single-Lane Bridge Problem][58]          │Younes Akhouayri             │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Software-Defined Perimeter][59]          │Luming Dong, Zhi Niu         │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Simplified Fast Paxos][60]               │Lim Ngian Xin Terry, Gaurav  │    │      │    │✔    │    
                                          │Gandhi                       │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Checkpoint Coordination][61]             │Andrew Helwer                │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Multi-Car Elevator System][62]           │Andrew Helwer                │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Nano Blockchain Protocol][63]            │Andrew Helwer                │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Readers-Writers Problem][64]         │Isaac DeFrain                │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Asynchronous Byzantine Consensus][65]    │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
                                          │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Folklore Reliable Broadcast][66]         │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
                                          │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Bosco Byzantine Consensus            │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
Algorithm][67]                            │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Consensus in One Communication Step][68] │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
                                          │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[One-Step Consensus with                  │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
Zero-Degradation][69]                     │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Failure Detector][70]                    │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
                                          │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Asynchronous Non-Blocking Atomic         │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
Commit][71]                               │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Asynchronous Non-Blocking Atomic         │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
Commitment with Failure Detectors][72]    │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Spanning Tree Broadcast Algorithm][73]   │Thanh Hai Tran, Igor Konnov, │    │      │    │✔    │    
                                          │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[The Cigarette Smokers Problem][74]       │Mariusz Ryndzionek           │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Conway's Game of Life][75]               │Mariusz Ryndzionek           │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Chameneos, a Concurrency Game][76]       │Mariusz Ryndzionek           │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[PCR Testing for Snippets of DNA][77]     │Martin Harrison              │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[RFC 3506: Voucher Transaction System][78]│Santhosh Raju, Cherry G.     │    │      │    │✔    │    
                                          │Mathew, Fransisca Andriani   │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Yo-Yo Leader Election][79]               │Ludovic Yvoz, Stephan Merz   │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[TCP as defined in RFC 9293][80]          │Markus Kuppe                 │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[B-trees][81]                             │Lorin Hochstein              │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[TLA⁺ Level Checking][82]                 │Leslie Lamport               │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Condition-Based Consensus][83]           │Thanh Hai Tran, Igor Konnov, │    │      │    │     │    
                                          │Josef Widder                 │    │      │    │     │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Buffered Random Access File][84]         │Calvin Loncaric              │    │      │    │✔    │    
──────────────────────────────────────────┼─────────────────────────────┼────┼──────┼────┼─────┼────
[Disruptor][85]                           │Nicholas Schultz-Møller      │    │      │    │✔    │    
──────────────────────────────────────────┴─────────────────────────────┴────┴──────┴────┴─────┴────

## Other Examples

Here is a list of specs stored in locations outside this repository or not validated by the CI, such
as submodules. Since these specs are not covered by CI testing it is possible they contain errors,
the reported details are incorrect, or they are no longer available. Ideally these will be moved
into this repo over time.

───────────────┬───────────────────────────────────────┬────────────────────────┬───┬───┬───┬───┬───
Spec           │Details                                │Author(s)               │Beg│TLA│TLC│Plu│Apa
               │                                       │                        │inn│PS │Mod│sCa│lac
               │                                       │                        │er │Pro│el │l  │he 
               │                                       │                        │   │of │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Blocking      │BlockingQueue                          │Markus Kuppe            │✔  │✔  │✔  │(✔)│   
Queue][86]     │                                       │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
IEEE 802.16    │2006, [paper][87], [specs][88]         │Prasad Narayana, Ruiming│   │   │   │   │   
WiMAX Protocols│                                       │Chen, Yao Zhao, Yan     │   │   │   │   │   
               │                                       │Chen, Zhi (Judy) Fu, Hai│   │   │   │   │   
               │                                       │Zhou                    │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
On the         │2016, [paper][89], [specs][90]         │Florent Chevrou, Aurélie│   │   │   │   │   
diversity of   │                                       │Hurault, Philippe       │   │   │   │   │   
asynchronous   │                                       │Quéinnec                │   │   │   │   │   
communication  │                                       │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Caesar][91]   │Multi-leader generalized consensus     │Giuliano Losa           │   │   │✔  │✔  │   
               │protocol (Arun et al., 2017)           │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[CASPaxos][92] │An extension of the single-decree Paxos│Tobias Schottdorf       │   │   │✔  │   │   
               │algorithm to a compare-and-swap type   │                        │   │   │   │   │   
               │register (Rystsov)                     │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[DataPort][93] │Dataport protocal 505.89PT, only PDF   │Geoffrey Biggs, Noriaki │   │   │   │   │   
               │files (Biggs & Noriaki, 2016)          │Ando                    │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[egalitarian-pa│Leaderless replication protocol based  │Iulian Moraru           │   │   │✔  │   │   
xos][94]       │on Paxos (Moraru et al., 2013)         │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[fastpaxos][95]│An extension of the classic Paxos      │Leslie Lamport          │   │   │   │   │   
               │algorithm, only PDF files (Lamport,    │                        │   │   │   │   │   
               │2006)                                  │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[fpaxos][96]   │A variant of Paxos with flexible       │Heidi Howard            │   │   │✔  │   │   
               │quorums (Howard et al., 2017)          │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[HLC][97]      │Hybrid logical clocks and hybrid vector│Murat Demirbas          │   │   │✔  │✔  │   
               │clocks (Demirbas et al., 2014)         │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[L1][98]       │Data center network L1 switch protocol,│Tom Rodeheffer          │   │   │   │   │   
               │only PDF files (Thacker)               │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[leaderless][99│Leaderless generalized-consensus       │Giuliano Losa           │   │   │✔  │✔  │   
]              │algorithms (Losa, 2016)                │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[losa_ap][100] │The assignment problem, a variant of   │Giuliano Losa           │   │   │✔  │✔  │   
               │the allocation problem                 │                        │   │   │   │   │   
               │(Delporte-Gallet, 2018)                │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[losa_rda][101]│Applying peculative linearizability to │Giuliano Losa           │   │   │   │   │   
               │fault-tolerant message-passing         │                        │   │   │   │   │   
               │algorithms and shared-memory consensus,│                        │   │   │   │   │   
               │only PDF files (Losa, 2014)            │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[m2paxos][102] │Multi-leader consensus protocols       │Giuliano Losa           │   │   │✔  │   │   
               │(Peluso et al., 2016)                  │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[mongo-repl-tla│A simplified part of Raft in MongoDB   │Siyuan Zhou             │   │   │✔  │   │   
][103]         │(Ongaro, 2014)                         │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[MultiPaxos][10│The abstract specification of          │Giuliano Losa           │   │   │✔  │   │   
4]             │Generalized Paxos (Lamport, 2004)      │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[naiad][105]   │Naiad clock protocol, only PDF files   │Tom Rodeheffer          │   │   │✔  │   │   
               │(Murray et al., 2013)                  │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[nfc04][106]   │Non-functional properties of           │Steffen Zschaler        │   │   │✔  │   │   
               │component-based software systems       │                        │   │   │   │   │   
               │(Zschaler, 2010)                       │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[raft][107]    │Raft consensus algorithm (Ongaro, 2014)│Diego Ongaro            │   │   │✔  │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[SnapshotIsolat│Serializable snapshot isolation (Cahill│Michael J. Cahill, Uwe  │   │   │✔  │   │   
ion][108]      │et al., 2010)                          │Röhm, Alan D. Fekete    │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[SyncConsensus]│Synchronized round consensus algorithm │Murat Demirbas          │   │   │✔  │✔  │   
[109]          │(Demirbas)                             │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Termination][1│Channel-counting algorithm (Kumar,     │Giuliano Losa           │   │✔  │✔  │✔  │✔  
10]            │1985)                                  │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Tla-tortoise-h│Robert Floyd's cycle detection         │Lorin Hochstein         │   │   │✔  │✔  │   
are][111]      │algorithm                              │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[VoldemortKV][1│Voldemort distributed key value store  │Murat Demirbas          │   │   │✔  │✔  │   
12]            │                                       │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Tencent-Paxos]│PaxosStore: high-availability storage  │Xingchen Yi, Hengfeng   │   │✔  │✔  │   │   
[113]          │made practical in WeChat. Proceedings  │Wei                     │   │   │   │   │   
               │of the VLDB Endowment(Zheng et al.,    │                        │   │   │   │   │   
               │2017)                                  │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Paxos][114]   │Paxos                                  │                        │   │   │✔  │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Lock-Free     │PlusCal spec of a lock-Free set used by│Markus Kuppe            │   │   │✔  │✔  │   
Set][115]      │TLC                                    │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[ParallelRaft][│A variant of Raft                      │Xiaosong Gu, Hengfeng   │   │   │✔  │   │   
116]           │                                       │Wei, Yu Huang           │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[CRDT-Bug][117]│CRDT algorithm with defect and fixed   │Alexander Niederbühl    │   │   │✔  │   │   
               │version                                │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[asyncio-lock][│Bugs from old versions of Python's     │Alexander Niederbühl    │   │   │✔  │   │   
118]           │asyncio lock                           │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Raft (with    │Raft with cluster changes, and a       │George Pîrlea, Darius   │   │   │✔  │   │✔  
cluster        │version with Apalache type annotations │Foom, Brandon Amos,     │   │   │   │   │   
changes)][119] │but no cluster changes                 │Huanchen Zhang, Daniel  │   │   │   │   │   
               │                                       │Ricketts                │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[MET for       │Model-check the CRDT designs, then     │Yuqi Zhang              │   │   │✔  │✔  │   
CRDT-Redis][120│generate test cases to test CRDT       │                        │   │   │   │   │   
]              │implementations                        │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Parallel      │Parallel threads incrementing a shared │Chris Jensen            │   │   │✔  │   │   
increment][121]│variable. Demonstrates invariants,     │                        │   │   │   │   │   
               │liveness, fairness and symmetry        │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[The Streamlet │Specification and model-checking of    │Giuliano Losa           │   │   │✔  │✔  │   
consensus      │both safety and liveness properties of │                        │   │   │   │   │   
algorithm][122]│Streamlet with TLC                     │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Petri         │Instantiable Petri Nets with liveness  │Eugene Huang            │   │   │✔  │   │   
Nets][123]     │properties                             │                        │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[CRDT][124]    │Specifying and Verifying CRDT Protocols│Ye Ji, Hengfeng Wei     │   │   │✔  │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Azure Cosmos  │Consistency models provided by Azure   │Dharma Shukla, Ailidani │   │   │✔  │✔  │   
DB][125]       │Cosmos DB                              │Ailijiang, Murat        │   │   │   │   │   
               │                                       │Demirbas, Markus Kuppe  │   │   │   │   │   
───────────────┼───────────────────────────────────────┼────────────────────────┼───┼───┼───┼───┼───
[Simple        │Spec of a microwave oven               │Konstantin Läufer,      │✔  │   │   │✔  │   
Microwave      │                                       │George K. Thiruvathukal │   │   │   │   │   
Oven][126]     │                                       │                        │   │   │   │   │   
───────────────┴───────────────────────────────────────┴────────────────────────┴───┴───┴───┴───┴───

## Contributing a Spec

See [`CONTRIBUTING.md`][127] for instructions.

## License

The repository is under the MIT license and you are encouraged to publish your spec under a
similarly-permissive license. However, your spec can be included in this repo along with your own
license if you wish.

## Support or Contact

Do you have any questions or comments? Please open an issue or send an email to the [TLA⁺ mailing
list][128].

[1]: https://gitpod.io/#https://github.com/tlaplus/examples/
[2]: https://github.com/tlaplus/Examples/actions/workflows/CI.yml
[3]: /tlaplus/Examples/blob/master/specifications
[4]: /tlaplus/Examples/blob/master/CONTRIBUTING.md
[5]: https://github.com/tlaplus/tlaplus
[6]: https://github.com/apalache-mc/apalache
[7]: https://github.com/tlaplus/tlapm
[8]: /tlaplus/Examples/blob/master/specifications/TeachingConcurrency
[9]: /tlaplus/Examples/blob/master/specifications/LoopInvariance
[10]: /tlaplus/Examples/blob/master/specifications/LearnProofs
[11]: /tlaplus/Examples/blob/master/specifications/Majority
[12]: /tlaplus/Examples/blob/master/specifications/sums_even
[13]: /tlaplus/Examples/blob/master/specifications/N-Queens
[14]: /tlaplus/Examples/blob/master/specifications/DiningPhilosophers
[15]: /tlaplus/Examples/blob/master/specifications/CarTalkPuzzle
[16]: /tlaplus/Examples/blob/master/specifications/DieHard
[17]: /tlaplus/Examples/blob/master/specifications/Prisoners
[18]: /tlaplus/Examples/blob/master/specifications/Prisoners_Single_Switch
[19]: /tlaplus/Examples/blob/master/specifications/SpecifyingSystems
[20]: /tlaplus/Examples/blob/master/specifications/tower_of_hanoi
[21]: /tlaplus/Examples/blob/master/specifications/MissionariesAndCannibals
[22]: /tlaplus/Examples/blob/master/specifications/Stones
[23]: /tlaplus/Examples/blob/master/specifications/CoffeeCan
[24]: /tlaplus/Examples/blob/master/specifications/ewd687a
[25]: /tlaplus/Examples/blob/master/specifications/Moving_Cat_Puzzle
[26]: /tlaplus/Examples/blob/master/specifications/Bakery-Boulangerie
[27]: /tlaplus/Examples/blob/master/specifications/MisraReachability
[28]: /tlaplus/Examples/blob/master/specifications/byzpaxos
[29]: /tlaplus/Examples/blob/master/specifications/barriers
[30]: /tlaplus/Examples/blob/master/specifications/locks_auxiliary_vars
[31]: /tlaplus/Examples/blob/master/specifications/ewd840
[32]: /tlaplus/Examples/blob/master/specifications/ewd998
[33]: /tlaplus/Examples/blob/master/specifications/Paxos
[34]: /tlaplus/Examples/blob/master/specifications/bcastByz
[35]: /tlaplus/Examples/blob/master/specifications/lamport_mutex
[36]: /tlaplus/Examples/blob/master/specifications/TwoPhase
[37]: /tlaplus/Examples/blob/master/specifications/PaxosHowToWinATuringAward
[38]: /tlaplus/Examples/blob/master/specifications/FiniteMonotonic
[39]: /tlaplus/Examples/blob/master/specifications/dijkstra-mutex
[40]: /tlaplus/Examples/blob/master/specifications/echo
[41]: /tlaplus/Examples/blob/master/specifications/TLC
[42]: /tlaplus/Examples/blob/master/specifications/transaction_commit
[43]: /tlaplus/Examples/blob/master/specifications/SlushProtocol
[44]: /tlaplus/Examples/blob/master/specifications/LeastCircularSubstring
[45]: /tlaplus/Examples/blob/master/specifications/KeyValueStore
[46]: /tlaplus/Examples/blob/master/specifications/chang_roberts
[47]: /tlaplus/Examples/blob/master/specifications/MultiPaxos-SMR
[48]: /tlaplus/Examples/blob/master/specifications/EinsteinRiddle
[49]: /tlaplus/Examples/blob/master/specifications/allocator
[50]: /tlaplus/Examples/blob/master/specifications/TransitiveClosure
[51]: /tlaplus/Examples/blob/master/specifications/acp
[52]: /tlaplus/Examples/blob/master/specifications/diskpaxos
[53]: /tlaplus/Examples/blob/master/specifications/SpanningTree
[54]: /tlaplus/Examples/blob/master/specifications/KnuthYao
[55]: /tlaplus/Examples/blob/master/specifications/Huang
[56]: /tlaplus/Examples/blob/master/specifications/ewd426
[57]: /tlaplus/Examples/blob/master/specifications/SlidingPuzzles
[58]: /tlaplus/Examples/blob/master/specifications/SingleLaneBridge
[59]: /tlaplus/Examples/blob/master/specifications/SDP_Verification
[60]: /tlaplus/Examples/blob/master/specifications/SimplifiedFastPaxos
[61]: /tlaplus/Examples/blob/master/specifications/CheckpointCoordination
[62]: /tlaplus/Examples/blob/master/specifications/MultiCarElevator
[63]: /tlaplus/Examples/blob/master/specifications/NanoBlockchain
[64]: /tlaplus/Examples/blob/master/specifications/ReadersWriters
[65]: /tlaplus/Examples/blob/master/specifications/aba-asyn-byz
[66]: /tlaplus/Examples/blob/master/specifications/bcastFolklore
[67]: /tlaplus/Examples/blob/master/specifications/bosco
[68]: /tlaplus/Examples/blob/master/specifications/c1cs
[69]: /tlaplus/Examples/blob/master/specifications/cf1s-folklore
[70]: /tlaplus/Examples/blob/master/specifications/detector_chan96
[71]: /tlaplus/Examples/blob/master/specifications/nbacc_ray97
[72]: /tlaplus/Examples/blob/master/specifications/nbacg_guer01
[73]: /tlaplus/Examples/blob/master/specifications/spanning
[74]: /tlaplus/Examples/blob/master/specifications/CigaretteSmokers
[75]: /tlaplus/Examples/blob/master/specifications/GameOfLife
[76]: /tlaplus/Examples/blob/master/specifications/Chameneos
[77]: /tlaplus/Examples/blob/master/specifications/glowingRaccoon
[78]: /tlaplus/Examples/blob/master/specifications/byihive
[79]: /tlaplus/Examples/blob/master/specifications/YoYo
[80]: /tlaplus/Examples/blob/master/specifications/tcp
[81]: /tlaplus/Examples/blob/master/specifications/btree
[82]: /tlaplus/Examples/blob/master/specifications/LevelChecking
[83]: /tlaplus/Examples/blob/master/specifications/cbc_max
[84]: /tlaplus/Examples/blob/master/specifications/braf
[85]: /tlaplus/Examples/blob/master/specifications/Disruptor
[86]: https://github.com/lemmy/BlockingQueue
[87]: https://users.cs.northwestern.edu/~ychen/Papers/npsec06.pdf
[88]: http://list.cs.northwestern.edu/802.16/
[89]: https://dl.acm.org/doi/10.1007/s00165-016-0379-x
[90]: http://hurault.perso.enseeiht.fr/asynchronousCommunication/
[91]: /tlaplus/Examples/blob/master/specifications/Caesar
[92]: /tlaplus/Examples/blob/master/specifications/CASPaxos
[93]: /tlaplus/Examples/blob/master/specifications/DataPort
[94]: /tlaplus/Examples/blob/master/specifications/egalitarian-paxos
[95]: /tlaplus/Examples/blob/master/specifications/fastpaxos
[96]: /tlaplus/Examples/blob/master/specifications/fpaxos
[97]: /tlaplus/Examples/blob/master/specifications/HLC
[98]: /tlaplus/Examples/blob/master/specifications/L1
[99]: /tlaplus/Examples/blob/master/specifications/leaderless
[100]: /tlaplus/Examples/blob/master/specifications/losa_ap
[101]: /tlaplus/Examples/blob/master/specifications/losa_rda
[102]: /tlaplus/Examples/blob/master/specifications/m2paxos
[103]: /tlaplus/Examples/blob/master/specifications/mongo-repl-tla
[104]: /tlaplus/Examples/blob/master/specifications/MultiPaxos
[105]: /tlaplus/Examples/blob/master/specifications/naiad
[106]: /tlaplus/Examples/blob/master/specifications/nfc04
[107]: /tlaplus/Examples/blob/master/specifications/raft
[108]: /tlaplus/Examples/blob/master/specifications/SnapshotIsolation
[109]: /tlaplus/Examples/blob/master/specifications/SyncConsensus
[110]: /tlaplus/Examples/blob/master/specifications/Termination
[111]: /tlaplus/Examples/blob/master/specifications/Tla-tortoise-hare
[112]: /tlaplus/Examples/blob/master/specifications/VoldemortKV
[113]: /tlaplus/Examples/blob/master/specifications/TencentPaxos
[114]: https://github.com/neoschizomer/Paxos
[115]: https://github.com/tlaplus/tlaplus/blob/master/tlatools/org.lamport.tlatools/src/tlc2/tool/fp
/OpenAddressing.tla
[116]: /tlaplus/Examples/blob/master/specifications/ParalleRaft
[117]: https://github.com/Alexander-N/tla-specs/tree/main/crdt-bug
[118]: https://github.com/Alexander-N/tla-specs/tree/main/asyncio-lock
[119]: https://github.com/dranov/raft-tla
[120]: https://github.com/elem-azar-unis/CRDT-Redis/tree/master/MET/TLA
[121]: https://github.com/Cjen1/tla_increment
[122]: https://github.com/nano-o/streamlet
[123]: https://github.com/elh/petri-tlaplus
[124]: https://github.com/JYwellin/CRDT-TLA
[125]: https://github.com/tlaplus/azure-cosmos-tla
[126]: /tlaplus/Examples/blob/master/specifications/microwave
[127]: /tlaplus/Examples/blob/master/CONTRIBUTING.md
[128]: https://groups.google.com/g/tlaplus
