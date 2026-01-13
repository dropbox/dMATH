# GenMC: A model checker for weak memory models

### Summary

GenMC is an open-source state-of-the-art model checker for verifying concurrent C/C++ programs under
the [RC11][1], [IMM][2], and LKMM memory models.

GenMC is based on a stateless model checking algorithm that is parametric in the choice of memory
model. Subject to a few basic conditions about the memory model, our algorithm is sound, complete
and optimal, in that it explores each consistent execution of the program according to the model
exactly once, and does not explore inconsistent executions or embark on futile exploration paths.

It incorporates many optimizations, such as lock-aware and barrier-aware partial order reduction,
symmetry reduction, and automatic spinloop bounding.

### Source distribution

* [GenMC on Github][3]

### Formal proofs

* [TruSt formalization in Coq][4]

### Tool papers

* GenMC: A model checker for weak memory models.
  Michalis Kokologiannakis and Viktor Vafeiadis.
  In CAV 2021 (July 2021)
  [[Paper (13 pages)]][5]
  [[Artifact @Zenodo]][6] [[Artifacts available]] [[Artifacts evaluated & functional]] [[Artifacts
  evaluated & reusable]]
* Enhancing GenMC's Usability and Performance.
  Michalis Kokologiannakis, Rupak Majumdar and Viktor Vafeiadis.
  In TACAS 2024 (April 2024)
  [[Paper (16 pages)]][7]
  [[Artifact @Zenodo]][8] [[Artifacts available]] [[Artifacts evaluated & reusable]]

### Papers

* Model checking for weakly consistent libraries.
  Michalis Kokologiannakis, Azalea Raad, and Viktor Vafeiadis.
  In PLDI 2019 (June 2019)
  [[Paper (15 pages)]][9] [[@ACM]][10] [[Full paper with the technical appendix (31 pages)]][11]
  [[Artifact @ACM]][12] [[Artifacts available]] [[Artifacts evaluated & functional]]
* Effective lock handling in stateless model checking.
  Michalis Kokologiannakis, Azalea Raad, and Viktor Vafeiadis.
  Proc. ACM Program. Lang. 3, OOPSLA, Article 173 (October 2019)
  [[Paper (26 pages)]][13] [[@ACM]][14] [[Full paper with the technical appendix (40 pages)]][15]
  [[Artifact @Zenodo]][16] [[Artifacts available]] [[Artifacts evaluated & functional]]
* HMC: Model checking for hardware memory models.
  Michalis Kokologiannakis and Viktor Vafeiadis.
  In ASPLOS 2020 (March 2020)
  [[Paper (15 pages)]][17] [[@ACM]][18]
  [[Artifact @Zenodo]][19] [[Artifacts available]] [[Artifacts evaluated & reusable]]
* BAM: Efficient model checking for barriers.
  Michalis Kokologiannakis and Viktor Vafeiadis.
  In NETYS 2021 (May 2021)
  [[Paper (17 pages)]][20]
* Dynamic partial order reductions for spinloops.
  Michalis Kokologiannakis, Xiaowei Ren, and Viktor Vafeiadis.
  In FMCAD 2021 (October 2021)
  [[Paper (10 pages)]][21]
* Truly stateless, optimal dynamic partial order reduction.
  Michalis Kokologiannakis, Iason Marmanis, Vladimir Gladstein, and Viktor Vafeiadis.
  Proc. ACM Program. Lang. 6, POPL (January 2022)
  [[Paper (28 pages)]][22]
  [[Full paper with the technical appendix (58 pages)]][23]
  [[Artifact @Zenodo]][24] [[Artifacts available]] [[Artifacts evaluated & reusable]]
* Model Checking on Multi-execution Memory Models.
  Evgenii Moiseenko, Michalis Kokologiannakis, and Viktor Vafeiadis.
  Proc. ACM Program. Lang. 6, OOPSLA2 (October 2022)
  [[Paper (28 pages)]][25]
  [[Full paper with the technical appendix (70 pages)]][26]
  [[Artifact @Zenodo]][27] [[Artifacts available]] [[Artifacts evaluated & reusable]]
* Reconciling Preemption Bounding with DPOR.
  Iason Marmanis, Michalis Kokologiannakis, and Viktor Vafeiadis.
  In TACAS 2023 (April 2023)
  [[Paper (19 pages)]][28]
  [[Full paper with the technical appendix (21 pages)]][29]
  [[Artifact @Zenodo]][30]
* Unblocking Dynamic Partial Order Reduction.
  Michalis Kokologiannakis, Iason Marmanis, and Viktor Vafeiadis.
  In CAV 2023 (July 2023)
  [[Paper (19 pages)]][31]
  [[Technical appendix]][32]
  [[Artifact @Zenodo]][33]
* SPORE: Combining Symmetry and Partial Order Reduction.
  Michalis Kokologiannakis, Iason Marmanis, and Viktor Vafeiadis.
  In PLDI 2024 (June 2024)
  [[Paper (23 pages)]][34]
  [[Full paper with the technical appendix]][35]
  [[Artifact @Zenodo]][36]
* RELINCHE: Automatically Checking Linearizability under Relaxed Memory Consistency.
  Pavel Golovin, Michalis Kokologiannakis, and Viktor Vafeiadis.
  In POPL 2025 (January 2025)
  [[Paper (23 pages)]][37]
  [[Artifact @Zenodo]][38]
* Model Checking C/C++ with Mixed-Size Accesses.
  Iason Marmanis, Michalis Kokologiannakis, and Viktor Vafeiadis.
  In POPL 2025 (January 2025)
  [[Paper (23 pages)]][39]
  [[Artifact @Zenodo]][40]

### People

* [Michalis Kokologiannakis][41] (MPI-SWS)
* [Iason Marmanis][42] (MPI-SWS)
* [Azalea Raad][43] (MPI-SWS)
* [Viktor Vafeiadis][44] (MPI-SWS)

### Related projects

* [RCMC][45]: a model checker for RC11
* [Repairing sequential consistency in C/C++11][46]: defines the RC11 model.
* [PerSeVerE][47]: GenMC for model checking under Linux ext4.
[Imprint][48] | [Data protection][49]

[1]: http://plv.mpi-sws.org/scfix/
[2]: http://plv.mpi-sws.org/imm/
[3]: https://www.github.com/mpi-sws/genmc/
[4]: https://github.com/volodeyka/trust-coq
[5]: cav21-paper.pdf
[6]: http://doi.org/10.5281/zenodo.4722966
[7]: tacas2024-gater.pdf
[8]: https://doi.org/10.5281/zenodo.10018135
[9]: paper.pdf
[10]: https://doi.org/10.1145/3314221.3314609
[11]: full-paper.pdf
[12]: https://dl.acm.org/do/10.1145/3325979/abs/
[13]: lapor-paper.pdf
[14]: https://doi.org/10.1145/3360599
[15]: lapor-full-paper.pdf
[16]: https://doi.org/10.5281/zenodo.3370296
[17]: hmc-paper.pdf
[18]: https://doi.org/10.1145/3373376.3378480
[19]: https://doi.org/10.5281/zenodo.3562082
[20]: netys2021-barriers.pdf
[21]: fmcad2021-spinloops.pdf
[22]: popl2022-trust.pdf
[23]: popl2022-trust-full.pdf
[24]: https://doi.org/10.5281/zenodo.5550765
[25]: oopsla2022-wmc.pdf
[26]: oopsla2022-wmc-full.pdf
[27]: https://doi.org/10.5281/zenodo.6821752
[28]: tacas23-paper.pdf
[29]: tacas23-full-paper.pdf
[30]: https://doi.org/10.5281/zenodo.7505917
[31]: cav2023-paper.pdf
[32]: cav2023-appendix.pdf
[33]: https://doi.org/10.5281/zenodo.7868370
[34]: pldi2024-spore.pdf
[35]: pldi2024-spore-full.pdf
[36]: https://zenodo.org/doi/10.5281/zenodo.10798179
[37]: popl2025-relinche.pdf
[38]: https://doi.org/10.5281/zenodo.13935829
[39]: popl2025-mixer.pdf
[40]: https://doi.org/10.5281/zenodo.13938749
[41]: https://people.mpi-sws.org/~michalis/
[42]: https://people.mpi-sws.org/~imarmanis/
[43]: https://people.mpi-sws.org/~azalea/
[44]: https://people.mpi-sws.org/~viktor/
[45]: http://plv.mpi-sws.org/rcmc/
[46]: http://plv.mpi-sws.org/scfix/
[47]: http://plv.mpi-sws.org/persevere/
[48]: https://imprint.mpi-klsb.mpg.de/sws/people/viktor
[49]: https://data-protection.mpi-klsb.mpg.de/sws/people/viktor
