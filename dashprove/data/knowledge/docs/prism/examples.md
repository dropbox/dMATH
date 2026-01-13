[[www.prismmodelchecker.org]][1]
[[P]][2]

* [Home][3]
* •
* [About][4]
  
  * » [About PRISM][5]
  * » [People][6]
  * » [Sponsors][7]
  * » [Contact][8]
  * » [News][9]
* •
* [Downloads][10]
  
  * » [PRISM][11]
  * » [PRISM-games][12]
  * » [Benchmarks][13]
  * » [GitHub][14]
  * » [Other downloads][15]
* •
* [Documentation][16]
  
  * » [Installation][17]
  * » [Manual][18]
  * » [FAQ][19]
  * » [Tutorial][20]
  * » [Lectures][21]
* •
* [Manual][22]
* •
* [Publications][23]
  
  * » [Selected publications][24]
  * » [PRISM publications][25]
  * » [PRISM bibliography][26]
  * » [External publications][27]
  * » [Search][28]
* •
* [Case Studies][29]
* •
* [Support][30]
  
  * » [Installation FAQ][31]
  * » [PRISM FAQ][32]
  * » [Forum (Google)][33]
* •
* [Developers][34]
  
  * » [GitHub][35]
  * » [Developer resources][36]
  * » [Developer forum][37]
* •
* [PRISM-games][38]
  
  * » [Download][39]
  * » [Publications][40]

# PRISM Case Studies

PRISM has been used to analyse a wide range of case studies in many different application domains. A
non-exhaustive list of these is given below. In many cases, we provide detailed information about
the case study, including PRISM language source code and experimental results. For others, we just
include links to the relevant publication.

We are always happy to include details of externally developed case studies. If you would like to
contribute content about your work with PRISM, or you want us to add a pointer to a publication
about your PRISM-related work, please [contact][41] us.

If you are interested in PRISM models for the purposes of benchmarking, see also the [PRISM
benchmark suite][42].

### Randomised distributed algorithms

These case studies examine the correctness and performance of various *randomised distributed
algorithms* taken from the literature.

* [Randomised self-stabilising algorithms][43] (Herman) (Israeli & Jalfon) (Beauquier et al.)
  [[KNP12a][44]]
* [Randomised two process wait-free test-and-set][45] (Tromp & Vitanyi)
* [Synchronous leader election protocol][46] (Itai & Rodeh)
* [Asynchronous leader election protocol][47] (Itai & Rodeh)
* [Randomised dining philosophers][48] (Lehmann & Rabin)
* [Randomised dining philosophers][49] (Lynch, Saias & Segala)
* [Dining cryptographers][50] (Chaum)
* [Randomised mutual exclusion][51] (Rabin)
* [Randomised mutual exclusion][52] (Pnueli & Zuck)
* [Randomised consensus protocol][53] (Aspnes & Herlihy) (with Cadence SMV and PRISM) [[KNS01a][54]]
* [Randomised consensus shared coin protocol][55] (Aspnes & Herlihy) (with PRISM) [[KNS01a][56]]
* [Byzantine agreement protocol][57] (Cachin, Kursawe & Shoup) (with Cadence SMV and PRISM)
  [[KN02][58]]
* [Byzantine agreement protocol][59] (Cachin, Kursawe & Shoup) (with PRISM) [[KN02][60]]
* [Rabin’s Choice Coordination Problem][61] [[NM10][62]]
  (contributed by Ukachukwu Ndukwu and Annabelle McIver)
* [Dice programs][63] (Knuth & Yao)
Other related case studies:

* Probabilistic leader election algorithms [[FP04][64]]
  (by [Wan Fokkink][65] and [Jun Pang][66])

### Communication, network and multimedia protocols

The following case studies investigate properties such as *quality of service* for (probabilistic)
*communication, network and multimedia protocols*.

* [Bluetooth device discovery][67] [[DKNP04][68], [DKNP06][69]] (with [Marie Duflot][70])
* [IEEE 802.3 CSMA/CD protocol][71] [[KNSW04][72], [KNSW07][73]]
* [Bounded retransmission protocol][74] (D'Argenio, Jeannet, Jensen & Larsen)
* [IEEE 1394 FireWire root contention protocol][75] (with PRISM) [[KNS03b][76]]
* [IEEE 1394 FireWire root contention protocol][77] (with KRONOS & PRISM) [[DKN02][78], [DKN04][79]]
* [IEEE 802.11 wireless LAN][80] [[KNS02a][81]]
* [IPv4 Zeroconf protocol][82] (Cheshire, Adoba & Guttman) [[KNPS06][83]]
* [IEEE 802.15.4 CSMA-CA protocol (ZigBee)][84] [[Fru06][85], [Fru11][86]] (Work with Matthias
  Fruth)
* [Probabilistic broadcast protocols][87] [[FG06][88]] (by [Ansgar Fehnker][89] and Peng Gao)
* [Gossip protocol][90] [[KNP08d][91]]
Other related case studies:

* Authenticated query flooding [[WS08][92]] (by Frank Werner and Peter Schmitt)
* ECo-MAC protocol for wireless sensor networks [[ZBB10][93]]
  (by Hafedh Zayani, Kamel Barkaoui and Rahma Ben Ayed)
* Congestion control in vehicular ad-hoc networks (VANETs) [[KF11][94]] (by Savas Konur and Michael
  Fisher)

### Security

These case studies use PRISM to analyse the correctness and performance of several
*security*-related systems.

Contract signing and fair exchange protocols:

* [Probabilistic contract signing][95] (Even, Goldreich & Lempel) [[NS06][96]] (with [Vitaly
  Shmatikov][97])
* [Probabilistic contract signing][98] (Ben-Or, Goldreich, Micali & Rivest) [[NS02][99],
  [NS06][100]] (with [Vitaly Shmatikov][101])
* [Probabilistic fair exchange][102] (Rabin) [[NS06][103]] (with [Vitaly Shmatikov][104])
* The ASW fair exchange protocol [[IZ08][105]] (by Salekul Islam and Mohammad Abu Zaid)

Anonymity:

* [Crowds protocol][106] (Reiter & Rubin) [[Shm02][107], [Shm04][108]] (contributed by [Vitaly
  Shmatikov][109])
* [Dining cryptographers][110] (Chaum)
* Crowds, Adithia, Onion routing and Tarzan [[Adi06][111]] (by Mariskha Adithia)
* Anonymity network topologies [[DSS04][112]] (by Roger Dingledine, [Vitaly Shmatikov][113] and Paul
  Syverson)

Threats and attacks:

* [PIN cracking schemes][114] [[Ste06][115]] (see also [this WIRED article][116]) (contributed by
  [Graham Steel][117])
* PIN block attacks [[Kal07][118]] (by Eirini Kaldeli)
* Quantification of Denial-of-Service (DoS) security threats [[BKPA08][119], [BKPA09][120]]
  (by Stylianos Basagiannis, Panagiotis Katsaros, Andrew Pombortsis and Nikolaos Alexiou)
* The Kaminsky DNS cache-poisoning attack [[ABK+10][121]] (see also [this page][122])
  (by Nikolaos Alexiou, Tushar Deshpande, Stylianos Basagiannis, Scott Smolka and Panagiotis
  Katsaros)
* The DNS bandwidth amplification attack [[DKBS11][123]]
  (by Tushar Deshpande, Panagiotis Katsaros, Stylianos Basagiannis and Scott Smolka)

Quantum cryptography protocols:

* BB84 [[Pap04][124], [GNP05][125]] (by [Rajagopal Nagarajan][126], [Nikos Papanikolaou][127] and
  [Simon Gay][128])
* B92 and BB84 [[EAA08][129], [EAA10a][130], [EAA10b][131]] (by Mohamed Elboukhari, Mostafa Azizi
  and Abdelmalek Azizi)
* BB84 [[FGHM11][132]] (by Verónica Fernández, María-José García-Martínez, Luis Hernández-Encinas
  and Agustín Martín)

Other security studies:

* [Non-repudiation protocol][133] (Markowitch & Roggeman) [[NPS13][134]]
* [Network Virus Infection][135] [[KNPV09][136]]
* Non-repudiation protocols
  [[LMST04][137], [Tro06][138]] (by Ruggero Lanotte, Andrea Maggiolo-Schettini and Angelo Troina)
  [[SM09][139]] (by Indranil Saha and Debapriyay Mukhopadhyay)
* A Reinforcement Model for Collaborative Security [[MS09][140]]
  (by Janardan Misra and Indranil Saha)
* A Certified E-Mail Protocol In Mobile Environments [[PBA+11][141], [BPA+11][142]]
  (by Stylianos Basagiannis, Sophia Petridou, Nikolaos Alexiou, Georgios Papadimitriou and
  Panagiotis Katsaros)
* The Needham-Schroeder (NS) and TMN protocols [[AA10][143]]
  (by Mojtaba Akbarzadeh and Mohammad Abdollahi Azgomi)
* SSL Handshake Protocol in Mobile Communications [[PB12][144]]
  (by Sophia Petridou and Stylianos Basagiannis)

### Biology

The following examples use PRISM to study the behaviour of various *biological* or
*biology-inspired* processes.

* [Cell cycle control in Eukaryotes][145] (Lecca and Priami)
* [FGF (Fibroblast Growth Factor) signalling][146] [[HKN+06][147], [HKN+08][148]] (with John Heath,
  Oksana Tymchyshyn)
* [MAPK cascade][149] (Huang and Ferrell) [[KNP08a][150]]
* DNA computing designs (with [Visual DSD][151] & PRISM) [[LPC+12][152]]
* [DNA walkers][153] [[DKTT13][154]]
* [Simple molecular reactions][155] (Shapiro)

The [PRISM tutorial][156] also includes a PRISM model of:

* A [circadian clock][157] (Barkai & Leiber)

Other related case studies:

* RKIP inhibited ERK pathway [[CVGO05][158], [CVGO06][159], [CGHV09][160]]
  (by [Muffy Calder][161], [Vladislav Vyshemirsky][162], [David Gilbert][163] and [Richard
  Orton][164])
* Codon bias [[PVBB07][165]] (by Tessa Pronk, Erik de Vink, Dragan Bosnacki and Timo Breit)
* Ribosome kinetics [[BPdV08][166]] (by Dragan Bosnacki, Tessa Pronk and Erik de Vink)
* Sorbitol dehydrogenase [[BCM+04][167], [BCM+05][168]]
  (by [Roberto Barbuti][169], [Stefano Cataudella][170], [Andrea Maggiolo-Schettini][171], [Paolo
  Milazzo][172] and [Angelo Troina][173])
* T Cell Signalling Events [[OTGT08][174]] (by Nick Owens, Jon Timmis, Andrew Greensted and Andy
  Tyrrell)
* Influenza virus fusion [[DDBM11][175]]
  (by Maria Pamela Dobay, Akos Dobay, Johnrob Bantang and Eduardo Mendoza)
* Platelet-Derived Growth Factor (PDGF) signalling [[WPM+11][176]]
  (by Qixia Yuan, Jun Pang, Sjouke Mauw, Panuwat Trairatphisan, Monique Wiesinger and Thomas Sauter)
* Bone pathologies [[LMP11][177]] (by Pietro Liò, Emanuela Merelli and Nicola Paoletti)

### Planning and synthesis

These case studies use PRISM to synthesise and/or analyse *strategies* and *controllers*:

* [Task Graph Scheduling][178] (Bouyer, Fahrenberg, Larsen & Markey) [[NPS13][179]]
* [Dynamic power management controllers][180] [[NPK+02][181], [NPK+03][182]]
  (with [Sandeep Shukla][183] and [Rajesh Gupta][184])
* [Human-in-the-loop UAV mission planning][185] [[FWHT15][186]]
  (with Lu Feng, Laura Humphrey and Ufuk Topcu)
* [Grid world robot][187] [[YKNP04][188], [YKNP06][189]]
  (with Håkan Younes)
* Robotic motion planning and control [[LWAB10][190]]
  (by M. Lahijanian, J. Wasniewski, S. B. Andersson, and C. Belta)
* Probabilistically safe vehicle control in a hostile environment [[CDLPB11][191]]
  (by Igor Cizelj, Xu Chu Ding, Morteza Lahijanian, Alessandro Pinto and Calin Belta)

### Game theory

The following case studies relate to or use *game-theoretic* concepts and techniques:

* [Microgrid demand side management][192] [[CFK+12][193], [CFK+13b][194]]
* [Stable matchings][195] [[BN12][196]] (with [Péter Biró][197])
* [Futures market investor][198] [[MM07][199], [MM02][200]]
  (with [Annabelle McIver][201] and [Carroll Morgan][202])
* [Alternating offers protocol][203] (Rubinstein) [[BFW06][204]]
  (contributed by Paolo Ballarini, Michael Fisher & Michael Wooldridge)
Other related case studies:

* Collective decision making for sensor networks [[CFK+12][205], [CFK+13b][206]]
* Trust models for user-centric networks [[KPS13][207]]

### Performance and reliability

These case studies consider *performance* and *reliability* properties of a variety of systems.

* [The thinkteam user interface][208] [[BML+04][209], [BML+05][210], [BML05a][211], [BML05b][212]]
  (contributed by Maurice ter Beek, Mieke Massink, Diego Latella, Stefania Gnesi, Alessandro
  Forghieri and Maurizio Sebastianis)
* [Embedded control system][213] (Muppala, Ciardo & Trivedi) [[KNP04c][214], [KNP07b][215]]
* [NAND multiplexing][216] [[NPKS05][217]]
  (with [Sandeep Shukla][218])
* [Workstation cluster][219] (Haverkort, Hermanns & Katoen)
* [Wireless communication cell][220] (Haring, Marie, Puigjaner & Trivedi)
* [Simple peer-to-peer protocol][221]
Other related case studies:

* Cloud computing live migration [[KM11][222]]
  (by Shinji Kikuchi and Yasuhide Matsumoto at Fujitsu)
* Publish-subscribe systems [[HBGS07][223]]
  (by Fei He, Luciano Baresi, Carlo Ghezzi and Paola Spoletini)
* Group membership protocols [[RSPV07][224]]
  (by Valério Rosset, Pedro F. Souto, Paulo Portugal and Francisco Vasques)
* Crossbar molecular switch memory [[CT08][225]]
  (by Ayodeji Coker and Valerie Taylor)
* Software architectures for multi-core platforms [[TK11][226]]
  (by Li Tan and Axel Krings)

### Power management

These case studies investigate the performance of several *power management systems*.

* [Dynamic power management controllers][227] [[NPK+02][228], [NPK+03][229]]
  (with [Sandeep Shukla][230] and [Rajesh Gupta][231])
* [Real-time dynamic voltage scaling][232] (Pillai & Shin) [[KNP05c][233]]
Other related case studies:

* Dynamic power management with two-priority request queues [[SDM08][234]]
  (by Aleksandra Sesic, Stanisa Dautovic and Veljko Malbasa)
* Environmentally powered wireless sensor nodes [[SAA+08][235]]
  (by Alexandru Susu, Andrea Acquaviva, David Atienza and Giovanni De Micheli)

### CTMC benchmarks

These examples are often used in the literature as *benchmarks* to study the efficiency of CTMC
analysis techniques. You can find a wider selection of benchmarks, for all models, in the [PRISM
benchmark suite][236].

* [Kanban system][237] (Ciardo & Tilgner)
* [Flexible manufacturing system][238] (Ciardo & Trivedi)
* [Cyclic server polling system][239] (Ibe & Trivedi)
* [Tandem queueing network][240] (Hermanns, Meyer-Kayser & Siegle)

### Miscellaneous

* [Random graphs][241]
  (with Michel de Rougemont)
* The Ising model [[STKT07][242], [STTK09][243]]
  (by Toshifusa Sekizawa, Tatsuhiro Tsuchiya, Tohru Kikuno and Koichi Takahashi)
* Cognitive assistive technology: The hand-washing problem [[Ma08][244]]
  (by Zhongdan Ma)
Site hosted at the Department of Computer Science, University of Oxford

### [Case Studies][245]

* [Randomised distributed algorithms][246]
* [Communication, network and multimedia protocols][247]
* [Security][248]
* [Biology][249]
* [Planning and synthesis][250]
* [Game-theory][251]
* [Performance and reliability][252]
* [Power management][253]
* [CTMC benchmarks][254]
* [Miscellaneous][255]

[1]: /
[2]: /
[3]: /
[4]: /about.php
[5]: /about.php
[6]: /people.php
[7]: /sponsors.php
[8]: /contact.php
[9]: /news.php
[10]: /download.php
[11]: /download.php
[12]: /games/download.php
[13]: /benchmarks/
[14]: https://github.com/prismmodelchecker
[15]: /other-downloads.php
[16]: /doc/
[17]: /manual/InstallingPRISM
[18]: /manual/
[19]: /manual/FrequentlyAskedQuestions
[20]: /tutorial/
[21]: /lectures/
[22]: /manual/
[23]: /publ-lists.php
[24]: /publ-selected.php
[25]: /publications.php
[26]: /bib.php
[27]: /bib-ext.php
[28]: /publ-search.php
[29]: /casestudies/index.php
[30]: /support.php
[31]: /manual/InstallingPRISM/CommonProblemsAndQuestions
[32]: /manual/FrequentlyAskedQuestions/
[33]: http://groups.google.com/group/prismmodelchecker
[34]: https://github.com/prismmodelchecker/prism/wiki
[35]: https://github.com/prismmodelchecker
[36]: https://github.com/prismmodelchecker/prism/wiki
[37]: http://groups.google.com/group/prismmodelchecker-dev
[38]: /games/
[39]: /games/download.php
[40]: /games/publ.php
[41]: /contact.php
[42]: /benchmarks/
[43]: self-stabilisation.php
[44]: /bibitem.php?key=KNP12a
[45]: test-and-set.php
[46]: synchronous_leader.php
[47]: asynchronous_leader.php
[48]: phil.php
[49]: phil_lss.php
[50]: dining_crypt.php
[51]: rabin.php
[52]: mutual.php
[53]: consensus.php
[54]: /bibitem.php?key=KNS01a
[55]: consensus_prism.php
[56]: /bibitem.php?key=KNS01a
[57]: byzantine.php
[58]: /bibitem.php?key=KN02
[59]: byzantine_prism.php
[60]: /bibitem.php?key=KN02
[61]: rabin_choice.php
[62]: /bibitem.php?key=NM10
[63]: dice.php
[64]: /bibitem.php?key=FP04
[65]: http://www.cs.vu.nl/~wanf/
[66]: http://www.lix.polytechnique.fr/~pangjun/
[67]: bluetooth.php
[68]: /bibitem.php?key=DKNP04
[69]: /bibitem.php?key=DKNP06
[70]: http://www.univ-paris12.fr/lacl/duflot/
[71]: csma.php
[72]: /bibitem.php?key=KNSW04
[73]: /bibitem.php?key=KNSW07
[74]: brp.php
[75]: firewire.php
[76]: /bibitem.php?key=KNS03b
[77]: firewire_kronos.php
[78]: /bibitem.php?key=DKN02
[79]: /bibitem.php?key=DKN04
[80]: wlan.php
[81]: /bibitem.php?key=KNS02a
[82]: zeroconf.php
[83]: /bibitem.php?key=KNPS06
[84]: zigbee.php
[85]: /bibitem.php?key=Fru06
[86]: /bibitem.php?key=Fru11
[87]: prob_broadcast.php
[88]: /bibitem.php?key=FG06
[89]: http://www.cse.unsw.edu.au/~ansgar/
[90]: gossip.php
[91]: /bibitem.php?key=KNP08d
[92]: /bibitem.php?key=WS08
[93]: /bibitem.php?key=ZBB10
[94]: /bibitem.php?key=KF11
[95]: contract_egl.php
[96]: /bibitem.php?key=NS06
[97]: http://www.cs.utexas.edu/~shmat/
[98]: contract.php
[99]: /bibitem.php?key=NS02
[100]: /bibitem.php?key=NS06
[101]: http://www.cs.utexas.edu/~shmat/
[102]: fairexchange.php
[103]: /bibitem.php?key=NS06
[104]: http://www.cs.utexas.edu/~shmat/
[105]: /bibitem.php?key=IZ08
[106]: crowds.php
[107]: /bibitem.php?key=Shm02
[108]: /bibitem.php?key=Shm04
[109]: http://www.cs.utexas.edu/~shmat/
[110]: dining_crypt.php
[111]: /bibitem.php?key=Adi06
[112]: /bibitem.php?key=DSS04
[113]: http://www.cs.utexas.edu/~shmat/
[114]: pincracking.php
[115]: /bibitem.php?key=Ste06
[116]: http://blog.wired.com/27bstroke6/2009/04/pins.html
[117]: http://homepages.inf.ed.ac.uk/gsteel/
[118]: /bibitem.php?key=Kal07
[119]: /bibitem.php?key=BKPA08
[120]: /bibitem.php?key=BKPA09
[121]: /bibitem.php?key=ABK+10
[122]: http://www.cs.sunysb.edu/~sas/kaminsky/
[123]: /bibitem.php?key=DKBS11
[124]: /bibitem.php?key=Pap04
[125]: /bibitem.php?key=GNP05
[126]: http://www.dcs.warwick.ac.uk/people/academic/Rajagopal.Nagarajan/
[127]: http://www.dcs.warwick.ac.uk/~nikos/
[128]: http://www.dcs.gla.ac.uk/~simon/
[129]: /bibitem.php?key=EAA08
[130]: /bibitem.php?key=EAA10a
[131]: /bibitem.php?key=EAA10b
[132]: /bibitem.php?key=FGHM11
[133]: nonrepudiation.php
[134]: /bibitem.php?key=NPS13
[135]: virus.php
[136]: /bibitem.php?key=KNPV09
[137]: /bibitem.php?key=LMST04
[138]: /bibitem.php?key=Tro06
[139]: /bibitem.php?key=SM09
[140]: /bibitem.php?key=MS09
[141]: /bibitem.php?key=PBA+11
[142]: /bibitem.php?key=BPA+11
[143]: /bibitem.php?key=AA10
[144]: /bibitem.php?key=PB12
[145]: cyclin.php
[146]: fgf.php
[147]: /bibitem.php?key=HKN+06
[148]: /bibitem.php?key=HKN+08
[149]: mapk_cascade.php
[150]: /bibitem.php?key=KNP08a
[151]: http://lepton.research.microsoft.com/webdna/
[152]: /bibitem.php?key=LPC+12
[153]: dna_walkers.php
[154]: /bibitem.php?key=DKTT13
[155]: molecules.php
[156]: /tutorial
[157]: /tutorial/circadian.php
[158]: /bibitem.php?key=CVGO05
[159]: /bibitem.php?key=CVGO06
[160]: /bibitem.php?key=CGHV09
[161]: http://www.dcs.gla.ac.uk/~muffy/
[162]: http://www.dcs.gla.ac.uk/~vlad/
[163]: http://www.dcs.gla.ac.uk/~drg/
[164]: http://www.dcs.gla.ac.uk/~rorton/
[165]: /bibitem.php?key=PVBB07
[166]: /bibitem.php?key=BPdV08
[167]: /bibitem.php?key=BCM+04
[168]: /bibitem.php?key=BCM+05
[169]: http://www.di.unipi.it/~barbuti/
[170]: http://www.di.unipi.it/~cataudel/
[171]: http://www.di.unipi.it/~maggiolo/
[172]: http://www.di.unipi.it/~milazzo/
[173]: http://www.di.unipi.it/~troina/
[174]: /bibitem.php?key=OTGT08
[175]: /bibitem.php?key=DDBM11
[176]: /bibitem.php?key=WPM+11
[177]: /bibitem.php?key=LMP11
[178]: task_graph.php
[179]: /bibitem.php?key=NPS13
[180]: power.php
[181]: /bibitem.php?key=NPK+02
[182]: /bibitem.php?key=NPK+03
[183]: http://www.ece.vt.edu/faculty/shukla.html
[184]: http://www.cecs.uci.edu/~rgupta/
[185]: human-uav.php
[186]: /bibitem.php?key=FWHT15
[187]: robot.php
[188]: /bibitem.php?key=YKNP04
[189]: /bibitem.php?key=YKNP06
[190]: /bibitem.php?key=LWAB10
[191]: /bibitem.php?key=CDLPB11
[192]: mdsm.php
[193]: /bibitem.php?key=CFK+12
[194]: /bibitem.php?key=CFK+13b
[195]: stable_matching.php
[196]: /bibitem.php?key=BN12
[197]: http://www.econ.core.hu/english/inst/biro.html
[198]: investor.php
[199]: /bibitem.php?key=MM07
[200]: /bibitem.php?key=MM02
[201]: http://www.ics.mq.edu.au/~anabel/
[202]: http://www.cse.unsw.edu.au/~carrollm/
[203]: negotiation.php
[204]: /bibitem.php?key=BFW06
[205]: /bibitem.php?key=CFK+12
[206]: /bibitem.php?key=CFK+13b
[207]: /bibitem.php?key=KPS13
[208]: thinkteam.php
[209]: /bibitem.php?key=BML+04
[210]: /bibitem.php?key=BML+05
[211]: /bibitem.php?key=BML05a
[212]: /bibitem.php?key=BML05b
[213]: embedded.php
[214]: /bibitem.php?key=KNP04c
[215]: /bibitem.php?key=KNP07b
[216]: nand.php
[217]: /bibitem.php?key=NPKS05
[218]: http://www.ecpe.vt.edu/faculty/Shukla.html
[219]: cluster.php
[220]: cell.php
[221]: peer2peer.php
[222]: /bibitem.php?key=KM11
[223]: /bibitem.php?key=HBGS07
[224]: /bibitem.php?key=RSPV07
[225]: /bibitem.php?key=CT08
[226]: /bibitem.php?key=TK11
[227]: power.php
[228]: /bibitem.php?key=NPK+02
[229]: /bibitem.php?key=NPK+03
[230]: http://www.ece.vt.edu/faculty/shukla.html
[231]: http://www.cecs.uci.edu/~rgupta/
[232]: voltage.php
[233]: /bibitem.php?key=KNP05c
[234]: /bibitem.php?key=SDM08
[235]: /bibitem.php?key=SAA+08
[236]: /benchmarks/
[237]: kanban.php
[238]: fms.php
[239]: polling.php
[240]: tandem.php
[241]: graph_connected.php
[242]: /bibitem.php?key=STKT07
[243]: /bibitem.php?key=STTK09
[244]: /bibitem.php?key=Ma08
[245]: /casestudies/index.php
[246]: /casestudies/index.php#randoalgs
[247]: /casestudies/index.php#commprotocols
[248]: /casestudies/index.php#security
[249]: /casestudies/index.php#biology
[250]: /casestudies/index.php#planning
[251]: /casestudies/index.php#gametheory
[252]: /casestudies/index.php#reliability
[253]: /casestudies/index.php#power
[254]: /casestudies/index.php#benchmarks
[255]: /casestudies/index.php#misc
