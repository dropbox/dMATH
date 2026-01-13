[SMACK] Software Verifier & Verification Toolchain [GitHub][1]

## About

SMACK is both a modular software verification toolchain and a self-contained software verifier. It
can be used to verify the assertions in its input programs. In its default mode, assertions are
verified up to a given bound on loop iterations and recursion depth; it contains experimental
support for unbounded verification as well. Under the hood, SMACK is a translator from the LLVM
compiler's popular intermediate representation (IR) into the Boogie intermediate verification
language (IVL). Sourcing LLVM IR exploits an increasing number of compiler front-ends,
optimizations, and analyses. Targeting Boogie exploits a canonical platform which simplifies the
implementation of algorithms for verification, model checking, and abstract interpretation.

## Publications

* [CANAL: A Cache Timing Analysis Framework via LLVM Transformation][2], Chungha Sung, Brandon
  Paulsen, Chao Wang, 33rd ACM/IEEE International Conference on Automated Software Engineering (ASE
  2018)
* [Formal Security Verification of Concurrent Firmware in SoCs using Instruction-Level Abstraction
  for Hardware][3], Bo-Yuan Huang, Sayak Ray, Aarti Gupta, Jason M. Fung, Sharad Malik, 55th Annual
  Design Automation Conference (DAC 2018)
* [Reducer-Based Construction of Conditional Verifiers][4], Dirk Beyer, Marie-Christine Jakobs,
  Thomas Lemberger, Heike Wehrheim, 40th International Conference on Software Engineering (ICSE
  2018)
* [ZEUS: Analyzing Safety of Smart Contracts][5], Sukrit Kalra, Seep Goel, Mohan Dhawan, Subodh
  Sharma, 25th Annual Network and Distributed System Security Symposium (NDSS 2018)
* [Formal Verification of Optimizing Compilers][6], Yiji Zhang, Lenore D. Zuck, Keynote at the 14th
  International Conference on Distributed Computing and Internet Technology (ICDCIT 2018)
* [Counterexample-Guided Bit-Precision Selection][7], Shaobo He, Zvonimir Rakamaric, 15th Asian
  Symposium on Programming Languages and Systems (APLAS 2017)
* [FaCT: A Flexible, Constant-Time Programming Language][8], Sunjay Cauligi, Gary Soeller, Fraser
  Brown, Brian Johannesmeyer, Yunlu Huang, Ranjit Jhala, Deian Stefan, IEEE Secure Development
  Conference (SecDev 2017)
* [Static Assertion Checking of Production Software with Angelic Verification][9], Shaobo He,
  Shuvendu Lahiri, Akash Lal, Zvonimir Rakamaric, 8th Workshop on Tools for Automatic Program
  Analysis (TAPAS 2017)
* [Refining Interprocedural Change-Impact Analysis using Equivalence Relations][10], Alex Gyori,
  Shuvendu Lahiri, Nimrod Partush, 26th ACM SIGSOFT International Symposium on Software Testing and
  Analysis (ISSTA 2017)
* [System Programming in Rust: Beyond Safety][11], Abhiram Balasubramanian, Marek S. Baranowski,
  Anton Burtsev, Aurojit Panda, Zvonimir Rakamaric, Leonid Ryzhyk, 16th Workshop on Hot Topics in
  Operating Systems (HotOS 2017)
* [Verifying Constant-Time Implementations][12], Jose Bacelar Almeida, Manuel Barbosa, Gilles
  Barthe, Francois Dupressoir, Michael Emmi, 25th USENIX Security Symposium (2016)
* [Statistical Similarity of Binaries][13], Yaniv David, Nimrod Partush, Eran Yahav, 37th ACM
  SIGPLAN Conference on Programming Language Design and Implementation (PLDI 2016)
* [SMACK Software Verification Toolchain][14], Montgomery Carter, Shaobo He, Jonathan Whitaker,
  Zvonimir Rakamaric, Michael Emmi, Demonstrations Track at the 38th IEEE/ACM International
  Conference on Software Engineering (ICSE 2016)
* [Fast and Precise Symbolic Analysis of Concurrency Bugs in Device Drivers][15], Pantazis
  Deligiannis, Alastair F. Donaldson, Zvonimir Rakamaric, 30th IEEE/ACM International Conference on
  Automated Software Engineering (ASE 2015)
* [SMACK+Corral: A Modular Verifier (Competition Contribution)][16], Arvind Haran, Montgomery
  Carter, Michael Emmi, Akash Lal, Shaz Qadeer, Zvonimir Rakamaric, 21st International Conference on
  Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2015)
* [ICE: A Robust Framework for Learning Invariants][17], Pranav Garg, Christof Löding, P.
  Madhusudan, Daniel Neider, 26th International Conference on Computer Aided Verification (CAV 2014)
* [SMACK: Decoupling Source Language Details from Verifier Implementations][18], Zvonimir Rakamaric,
  Michael Emmi, 26th International Conference on Computer Aided Verification (CAV 2014) [MAIN
  REFERENCE]
* [Modular Verification of Shared-Memory Concurrent System Software][19], Zvonimir Rakamaric, Ph.D.
  Thesis, Department of Computer Science, University of British Columbia, 2011
* [A Scalable Memory Model for Low-Level Code][20], Zvonimir Rakamaric, Alan J. Hu, 10th
  International Conference on Verification, Model Checking and Abstract Interpretation (VMCAI 2009)
* [Automatic Inference of Frame Axioms Using Static Analysis][21], Zvonimir Rakamaric, Alan J. Hu,
  23rd IEEE/ACM International Conference on Automated Software Engineering (ASE 2008)

## People

## Coordinators

[[Michael Emmi]][22]
Michael Emmi
[[Zvonimir Rakamaric]][23]
Zvonimir Rakamaric

## Contributors

[[Montgomery Carter]][24]
Montgomery Carter
[[Pantazis Deligiannis]][25]
Pantazis Deligiannis
[[Arvind Haran]][26]
Arvind Haran
[[Shaobo He]][27]
Shaobo He
[[Jiten Thakkar]][28]
Jiten Thakkar
[[Jonathan Whitaker]][29]
Jonathan Whitaker
[[Marek Baranowski]][30]
Marek Baranowski
[[Jack J. Garzella]][31]
Jack J. Garzella
[[Liam Machado]][32]
Liam Machado
[[Dietrich Geisler]][33]
Dietrich Geisler

[1]: https://github.com/smackers/smack
[2]: https://dl.acm.org/citation.cfm?id=3240485
[3]: https://dl.acm.org/citation.cfm?id=3196055
[4]: https://dl.acm.org/citation.cfm?id=3180259
[5]: https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_09-1_Kalra_paper.pdf
[6]: https://link.springer.com/chapter/10.1007/978-3-319-72344-0_3
[7]: https://soarlab.org/publications/2017_aplas_hr
[8]: https://cseweb.ucsd.edu/~dstefan/pubs/cauligi:2017:fact.pdf
[9]: https://soarlab.org/publications/2017_tapas_hllr
[10]: https://www.microsoft.com/en-us/research/publication/refining-interprocedural-change-impact-an
alysis-using-equivalence-relations
[11]: https://soarlab.org/publications/2017_hotos_bbbprr
[12]: https://www.usenix.org/conference/usenixsecurity16/technical-sessions/presentation/almeida
[13]: http://dl.acm.org/citation.cfm?id=2908126
[14]: https://soarlab.org/publications/2016_icse_chwre
[15]: https://soarlab.org/publications/2015_ase_ddr
[16]: https://soarlab.org/publications/2015_tacas_hcelqr
[17]: http://madhu.cs.illinois.edu/CAV14ice.pdf
[18]: https://soarlab.org/publications/2014_cav_re
[19]: https://soarlab.org/publications/2011_thesis_rakamaric
[20]: https://soarlab.org/publications/2009_vmcai_rh
[21]: https://soarlab.org/publications/2008_ase_rh
[22]: http://michael-emmi.github.io/
[23]: http://www.zvonimir.info/
[24]: http://www.linkedin.com/pub/montgomery-carter/12/a89/512
[25]: http://pdeligia.github.io/
[26]: http://www.cs.utah.edu/~haran
[27]: http://www.cs.utah.edu/~shaobo/
[28]: http://jiten-thakkar.com/
[29]: https://www.linkedin.com/in/jonathan-whitaker-5a8b2484
[30]: https://github.com/keram88
[31]: https://www.linkedin.com/in/jack-j-garzella-7140a716
[32]: https://github.com/liammachado
[33]: https://www.linkedin.com/in/dietrich-geisler-999204133
