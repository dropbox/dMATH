### Documentation

**New [tutorial for CPAchecker 3.0][1] available!**
Covers installation, first steps, and how to use different analyses.
This is an extended version of a [publication at FM 2024][2]. A [recording of the tutorial at FM
2024][3] is now also available.


#### Further links for more information on how to use CPAchecker

* [Instructions for installation][4]
  If you have problems with the installation, please contact us (either via [mailing list][5] or
  Dirk).
* [Getting started with CPAchecker][6]
* [More tutorials on specific aspects of CPAchecker][7]
* [Documentation overview...][8]
* [Counterexample report][9] for an [example program][10] with a bug found by CPAchecker
* [Project statistics at OpenHUB][11]

#### Understanding CPAchecker

* Basic Architecture and Concepts
  
  * [Slides][12]
  * [CPAchecker: A Tool for Configurable Software Verification][13]. CAV 2011
  * [Configurable Software Verification][14]. CAV 2007
  * [Program Analysis with Dynamic Precision Adjustment][15]. ASE 2008
  * [Combining Model Checking and Data-Flow Analysis][16]. Handbook on Model Checking 2018
* Predicate Analysis
  
  * [Slides: A Unifying View on SMT-Based Software Verification][17]
  * [Predicate Abstraction with Adjustable-Block Encoding][18]. FMCAD 2010
  * [A Unifying View on SMT-Based Software Verification][19], JAR 2018
* [Explicit-State Software Model Checking Based on CEGAR and Interpolation][20]. FASE 2013

### Possible Projects

A selection of available topics can be found [here][21]. We have prepared a specific list of [ideas
that would be suitable for a Google Summer of Code project][22]. We also encourage developers to
bring new ideas and implement new approches.

### CPAchecker Workshops

Since 2016 a yearly workshop is organized for users and developers of CPAchecker:

* [10th International Workshop on CPAchecker (CPA '25)][23]: June 28, 2025 in Trondheim, Norway,
  cohosted with [ISSTA '25][24] ([program and slides][25]).
* [9th International Workshop on CPAchecker (CPA '24)][26]: September 09, 2024 in Milano, Italy,
  cohosted with [FM '24][27] ([program and slides][28]).
* [8th International Workshop on CPAchecker (CPA '23)][29]: September 11, 2023 in Kirchberg,
  Luxembourg, cohosted with [ASE '23][30] ([program and slides][31]).
* [7th International Workshop on CPAchecker (CPA '22)][32]: October 5-6, 2022 in Oldenburg, Germany
  ([program and slides][33]).
* [6th International Workshop on CPAchecker (CPA '21)][34]: September 30-October 1, 2021 online
  ([program, slides, and videos][35]).
* [5th International Workshop on CPAchecker (CPA '20)][36]: September 28-29, 2020 online ([program
  and slides][37])
* [4th International Workshop on CPAchecker (CPA '19)][38]: October 1-2, 2019 in Frauenchiemsee,
  Germany ([program and slides][39])
* [3rd International Workshop on CPAchecker (CPA '18)][40]: September 25-26, 2018 in Moscow, Russia,
  together with the 8th Linux Driver Verification (LDV) Workshop ([program and slides][41])
* [2nd International Workshop on CPAchecker (CPA '17)][42]: September 4-5, 2017 in Paderborn,
  Germany ([program and slides][43])
* [1st International Workshop on CPAchecker (CPA '16)][44]: September 22-23, 2016 in Passau, Germany
  ([program and slides][45])

[1]: https://arxiv.org/pdf/2409.02094
[2]: https://doi.org/10.1007/978-3-031-71177-0_30
[3]: https://www.youtube.com/watch?v=l9n4KvbrvqY
[4]: https://gitlab.com/sosy-lab/software/cpachecker/blob/trunk/INSTALL.md
[5]: https://groups.google.com/forum/#!forum/cpachecker-users
[6]: https://gitlab.com/sosy-lab/software/cpachecker/blob/trunk/README.md
[7]: https://gitlab.com/sosy-lab/software/cpachecker/tree/trunk/doc/tutorials
[8]: https://gitlab.com/sosy-lab/software/cpachecker/tree/trunk/doc
[9]: counterexample-report/ErrorPath.0.html
[10]: https://github.com/sosy-lab/sv-benchmarks/blob/8a287a460a1b2f817f36d02b2e22516f9b0a9a6e/c/loop
s/bubble_sort_false-unreach-call.i
[11]: https://www.openhub.net/p/cpachecker
[12]: https://www.sosy-lab.org/research/prs/Latest_CPAchecker.pdf
[13]: https://www.sosy-lab.org/research/pub/2011-CAV.CPAchecker_A_Tool_for_Configurable_Software_Ver
ification.pdf
[14]: https://www.sosy-lab.org/research/pub/2007-CAV.Configurable_Software_Verification.pdf
[15]: https://www.sosy-lab.org/research/pub/2008-ASE.Program_Analysis_with_Dynamic_Precision_Adjustm
ent.pdf
[16]: https://www.sosy-lab.org/research/pub/2018-HBMC.Combining_Model_Checking_and_Data-Flow_Analysi
s.pdf
[17]: https://www.sosy-lab.org/research/prs/Latest_UnifyingViewSmtBasedSoftwareVerification.pdf
[18]: https://www.sosy-lab.org/research/pub/2010-FMCAD.Predicate_Abstraction_with_Adjustable-Block_E
ncoding.pdf
[19]: https://www.sosy-lab.org/research/pub/2018-JAR.A_Unifying_View_on_SMT-Based_Software_Verificat
ion.pdf
[20]: https://www.sosy-lab.org/research/pub/2013-FASE.Explicit-State_Software_Model_Checking_Based_o
n_CEGAR_and_Interpolation.pdf
[21]: https://www.sosy-lab.org/teaching.php#thesis
[22]: https://www.sosy-lab.org/gsoc/gsoc2019.php
[23]: https://cpa.sosy-lab.org/2025/
[24]: https://conf.researchr.org/home/issta-2025
[25]: https://cpa.sosy-lab.org/2025/program.html
[26]: https://cpa.sosy-lab.org/2024/
[27]: https://www.fm24.polimi.it/
[28]: https://cpa.sosy-lab.org/2024/program.html
[29]: https://cpa.sosy-lab.org/2023/
[30]: https://conf.researchr.org/home/ase-2023
[31]: https://cpa.sosy-lab.org/2023/program.html
[32]: https://uol.de/en/computingscience/groups/formal-methods/forschung/7th-international-workshop-
on-cpachecker-cpa-22
[33]: https://uol.de/en/computingscience/groups/formal-methods/forschung/7th-international-workshop-
on-cpachecker-cpa-22/workshop-program
[34]: https://cpa.sosy-lab.org/2021/
[35]: https://cpa.sosy-lab.org/2021/program.html
[36]: https://www.informatik.tu-darmstadt.de/svpsys/semantik_und_verifikation_paralleler_systeme_svp
sys/svpsys_cpa2020.en.jsp
[37]: https://www.informatik.tu-darmstadt.de/svpsys/semantik_und_verifikation_paralleler_systeme_svp
sys/svpsys_cpa2020.en.jsp
[38]: https://cpa.sosy-lab.org/2019/
[39]: https://cpa.sosy-lab.org/2019/program.php
[40]: http://linuxtesting.org/ldv-cpa-workshop-2018
[41]: http://linuxtesting.org/ldv-cpa-workshop-2018
[42]: https://cs.uni-paderborn.de/sms/cpa17-2nd-international-workshop-on-cpachecker-cpa17/
[43]: https://cs.uni-paderborn.de/sms/cpa17-2nd-international-workshop-on-cpachecker-cpa17/program/
[44]: https://cpa16.sosy-lab.org/
[45]: https://cpa16.sosy-lab.org/program.php
