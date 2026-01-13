**
[SeaHorn][1]
Toggle navigation

* [Blog][2]
* [About][3]
* [Techniques][4]
* [Download][5]
* [Publications][6]
* [People][7]
[Fork me on GitHub][8]

# SeaHorn

A fully automated analysis framework for LLVM-based languages.

# In Action


# About

#### [ Front-End ][9]

Takes an LLVM based program (e.g., C) input program and generates LLVM IR bitcode. Specifically, it
performs the pre-processing and optimization of the bitcode for verification purposes.

#### [ Middle-End ][10]

Takes as input the optimized LLVM bitcode and emits verification condition as Constrained Horn
Clauses (CHC). The middle-end is in charge of selecting encoding of the VCs and the degree of
precision.

#### [ Back-End ][11]

Takes CHC as input and outputs the result of the analysis. In principle, any verification engine
that digests CHC clauses could be used to discharge the VCs. Currently, SeaHorn employs several
SMT-based model checking engines based on PDR/IC3. Complementary, SeaHorn uses the abstract
interpretation-based analyzer [CRAB][12] (A language-agnostic framework for abstract interpretation)
for providing numerical invariants.

# Analysis techniques

(The way we prove things)

** (Non)-Termination checking
(Non)-termination checking
×Close

#### (Non)-Termination checking

SeaHorn can be used to prove termination and non-termination of programs. The core of the approach
lies on the innovative use of the backend verification engines to build termination arguments. More
specifically, the backend verification engines are used to systematically sample terminating program
execution and to extrapolate from these a candidate ranking function for the program. The candidate
ranking function can be a piecewise-defined, lexicographic, or multiphase combination of affine
functions of the program variables. The backend verification engines are then used to validate the
candidate ranking function, or to otherwise provide a witness for program non-termination. More
information about this can be found this [paper][13].

Close
** Inconsistency Checking
Inconsistecncy Checking
×Close

#### Inconsistency Checking

Inconsistent code is an important class of program abnormalities that appears in real-world code
bases and often reveals serious bugs. A piece of code is inconsistent if it is not part of any
safely terminating execution. Existing approaches to inconsistent code detection scale to programs
with millions of lines of code, and have lead to patches in applications like the web-server Tomcat
or the Linux kernel. However, the ability of existing tools to detect inconsistencies is limited by
gross over-approximation of looping control-flow. SeaHorn is able to detect inconsistent code
fragments. This is done by computing a path coverage for the generated system of Horn clauses. More
information about this can be found in this [paper][14].

Close

# Publications

* ** S. Wesley, M. Christakis, J. A. Navas, R. J. Trefler, V. Wüstholz, and A. Gurfinkel. Verifying
  Solidity Smart Contracts Via Communication Abstraction in SmartACE. At VMCAI 2022.
* ** S. Priya, X. Zhou, Y. Su, Y. Vizel, Yuyan Bao, and A. Gurfinkel. Verifying Verified Code. At
  ATVA 2021.
* ** S. Wesley, M. Christakis, J. A. Navas, R. J. Trefler, V. Wüstholz, and A. Gurfinkel.
  Compositional Verification of Smart Contracts Through Communication Abstraction. At SAS 2021.
* ** A. Gurkinkel and J.A. Navas. Abstract Interpretation of LLVM with a Region-Based Memory Model.
  At VSTTE 2021.
* ** J. A. Navas, B. Dutertre, I. Mason. Verification of an Optimized NTT Algorithm. At VSTTE 2020.
* **J. Kuderski, A. Gurfinkel and J.A.Navas. Unification-based Pointer Analysis without Oversharing.
  At FMCAD 2019.
* **E. Gershuni, N. Amit, A. Gurfinkel, N. Narodytska, J. A. Navas, N. Rinetzky, L. Ryzhyk and M.
  Sagiv. [ Simple and Precise Static Analysis of Untrusted Linux Kernel Extensions ][15].
  ([Slides][16]) At PLDI 2019.
* **J. Gennari, A. Gurfinkel, T.Kahsai, J.A.Navas, and E.J.Schwartz. [ Executable Counterexamples in
  Software Model Checking ][17]. ([Slides][18]) At VSTTE 2018.
* **A. Gurfinkel and J.A.Navas. [ A Context-Sensitive Memory Model for Verification of C/C++
  Programs ][19]. ([Slides][20]) At SAS 2017.
* **A. Gurfinkel: [ SeaHorn Tutorial][21].
* **C. Urban, A. Gurfinkel, T.Kahsai. [Synthesizing Ranking Functions from Bits and Pieces][22]. At
  TACAS 2016.
* **T.Kahsai, J.A.Navas, D. Jovanovic, M. Schaf. [ Finding Inconsistencies in Programs with
  Loops][23]. At LPAR 2015. LNCS 9450, pp. 499-514, 2015.
* **G.Gange, J.A.Navas, P.Schachte, H.Sondergaard, P.Stuckey. [ Exploiting Sparsity in
  Difference-Bound Matrices ][24]. At SAS 2016.
* **G.Gange, J.A.Navas, P.Schachte, H.Sondergaard, P.Stuckey. [ An Abstract Domain of Uninterpreted
  Functions][25]. At VMCAI 2016.
* **A. Gurfinkel [ Algoritmic Logic-Based Verification with SeaHorn][26]. Invited tutorial at SYNASC
  2015.
* **A. Komuravelli, N.Bjorner, A. Gurfinkel, K. McMillan. *Compositional Verification of Procedural
  Programs using Horn Clauses over Integers and Arrays*. At FMCAD 2015. IEEE, pp. 89-96, 2015.
* **A.Gurfinkel, T.Kahsai, A. Komuravelli, J.A.Navas. [ The SeaHorn Verification Framework][27]. At
  CAV 2015. LNCS 9206, pp. 343-361. 2015
* **A.Gurfinkel, T.Kahsai, J.A.Navas. [Algorithmic Logic-Based Verification][28]. At ACM SIGLOG
  Newsletter, April 2015.
* **A.Gurfinkel, T.Kahsai, J.A.Navas. [SeaHorn: A Framework for Verifying C Programs (Competition
  Contribution)][29]. At TACAS 2015. LNCS 9035. pp 447-450. 2015
* **A. Komuravelli, A.Gurfinkel, S.Chaki. [ SMT-Based Model Checking for Recursive Programs.][30].
  At CAV 2014. LNCS 8559, pp. 17-34. 2014

# Download

* [ SeaHorn Docker Image][31]

# Founders

* [Arie Gurfinkel
  ][32][(UWaterloo)][33]
* [Temesghen Kahsai
  (Amazon)][34]
* [Jorge Navas
  (Certora)][35]

© Arie Gurfinkel 2018| [Bootstrap][36] | [Worthy][37].

[1]: #banner
[2]: http://seahorn.github.io/blog/
[3]: #about
[4]: #tec
[5]: #download
[6]: #publications
[7]: #people
[8]: https://github.com/seahorn/seahorn
[9]: #collapseOne
[10]: #collapseTwo
[11]: #collapseThree
[12]: https://github.com/seahorn/crab
[13]: papers/termination_tacas16.pdf
[14]: papers/lpar2015.pdf
[15]: papers/ebpf-pldi19.pdf
[16]: papers/EBPF-PLDI-19.pptx
[17]: papers/cex-vstte18.pdf
[18]: papers/cex-vstte18-slides.pdf
[19]: papers/sea-dsa-SAS17.pdf
[20]: papers/sas17_slides.pdf
[21]: https://arieg.bitbucket.io/ssft15.html
[22]: papers/termination_tacas16.pdf
[23]: papers/lpar2015.pdf
[24]: papers/split-dbm-sas16.pdf
[25]: papers/vmcai16.pdf
[26]: https://arieg.bitbucket.io/pdf/seahorn_synasc15.pdf
[27]: papers/cav15.pdf
[28]: http://siglog.hosting.acm.org/wp-content/uploads/2015/04/siglog_news_4.pdf
[29]: http://clip.dia.fi.upm.es/~jorge/docs/seahorn-svcomp15.pdf
[30]: papers/spacer2014.pdf
[31]: https://hub.docker.com/r/seahorn/seahorn/
[32]: https://arieg.bitbucket.io
[33]: http://ece.uwaterloo.ca
[34]: http://lememta.info
[35]: https://jorgenavas.github.io/
[36]: http://twitter.github.io/bootstrap
[37]: http://htmlcoder.me
