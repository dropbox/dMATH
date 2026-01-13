On this page, you can find (extended) benchmark results accompanying paper submissions.

## QComp 2020 [[1]][1]

Storm participated in the *2020 Comparison of Tools for the Analysis of Quantitative Formal Models
(QComp 2020)*. Storm participated in two modes: with the new *automatic* engine selection and with
*static* engine selection.

Result summary for the *ε-correct* track:

[Article]

Result summary for the *often ε-correct* track:

[Article]

Further results for all tracks can be found in the [competition report][2] and the detailed result
tables on the [QComp 2020 website][3].

## The Probabilistic Model Checker Storm [[2]][4]

#### Setup

The benchmarks were run on 4 cores of an Intel Xeon Platinum 8160 Processor with 12GB of memory
available. The timeout was 1800 seconds.

#### Benchmarks

All benchmarks from [QComp 2019][5] were considered, except for 4 PTA Benchmarks that were not
compatible with Storm.

#### Results

[Show interactive result table][6] [Download raw data and replication package][7]

## QComp 2019 [[3]][8]

Storm participated in the *2019 Comparison of Tools for the Analysis of Quantitative Formal Models
(QComp 2019)*. Details on the competition, the participating tools and the QComp benchmark set can
be found on the [competition website][9]. Detailed results are available in the [interactive results
table][10].

## A Storm is Coming: A Modern Probabilistic Model Checker [[4]][11]

#### Setup

The benchmarks were conducted on a HP BL685C G7. All tools had up to eight cores with 2.0GHz and 8GB
of memory available, but only the Java garbage collection of PRISM and EPMC used more than one core.
The timeout was set to 1800 seconds.

#### Benchmarks

In this paper, we use all non-PTA models from the [PRISM benchmark suite][12] and the [IMCA][13]
benchmarks for MAs.

#### Results

In order to share the results, we provide them as a set of log files. To make the results more
accessible, we also give four tables (one for each model type: DTMC, CTMC, MDP and MA). Using the
buttons near the top of the table, you can select which of the configurations of the tools are
displayed side-by-side (by default all configurations of participating tools are shown). For
example, clicking `Storm-sparse` toggles whether Storm’s sparse engine participates in the
comparison or not. As argued in the paper, it makes sense to compare “matching engines” of the
tools. (For an explanation which engines are comparable, please consult the paper.) This is why
there are also buttons that let you select the tool configurations that are comparable with one
click (`Show sparse`, `Show hybrid`, `Show dd` and `Show exact`). The best time for each instance is
marked green. By clicking on the time for a particular experiment, you are taken to the log file of
that experiment.

The log files were obtained using a version of Storm that represents “infinity” as “-1” in `--exact`
mode. This is, however just a displaying issue. Newer versions of Storm correctly display “inf” as
the result.

[Show DTMC table][14] [Show CTMC table][15] [Show MDP table][16] [Show MA table][17] [Show log
files][18]

## References

* Carlos E. Budde *et al.*, “On Correctness, Precision, and Performance in Quantitative Verification
  - QComp 2020 Competition Report,” in *ISoLA (4)*, 2020.
  [ [DOI: 10.1007/978-3-030-83723-5_15] ][19] [ [further information] ][20]
* Christian Hensel, Sebastian Junges, Joost-Pieter Katoen, Tim Quatmann, and Matthias Volk, “The
  probabilistic model checker Storm,” *Int. J. Softw. Tools Technol. Transf.*, no. 4, 2022.
  [ [DOI: 10.1007/s10009-021-00633-z] ][21] [ [arXiv: 2002.07080] ][22] [ [supplemental material]
  ][23] [ [further information] ][24]
* Ernst Moritz Hahn *et al.*, “The 2019 Comparison of Tools for the Analysis of Quantitative Formal
  Models - (QComp 2019 Competition Report),” in *TACAS (3)*, 2019.
  [ [DOI: 10.1007/978-3-030-17502-3_5] ][25] [ [supplemental material] ][26] [ [further information]
  ][27]
* Christian Dehnert, Sebastian Junges, Joost-Pieter Katoen, and Matthias Volk, “A Storm is Coming: A
  Modern Probabilistic Model Checker,” in *CAV (2)*, 2017.
  [ [DOI: 10.1007/978-3-319-63390-9_31] ][28] [ [arXiv: 1702.04311] ][29] [ [further information]
  ][30]

[1]: #BHKKPQTZ20
[2]: https://doi.org/10.1007/978-3-030-83723-5_15
[3]: https://qcomp.org/competition/2020/
[4]: #HJKQV22
[5]: https://qcomp.org/competition/2019/index.html
[6]: https://moves-rwth.github.io/storm-benchmark-logs/docs/2020-09/table.html
[7]: https://doi.org/10.5281/zenodo.4017717
[8]: #HHHKKKPQRS19
[9]: https://qcomp.org/competition/2019/index.html
[10]: https://qcomp.org/competition/2019/results/index.html
[11]: #DJKV17
[12]: https://www.prismmodelchecker.org/benchmarks/
[13]: https://github.com/buschko/imca
[14]: https://moves-rwth.github.io/storm-benchmark-logs/docs/index_dtmc.html
[15]: https://moves-rwth.github.io/storm-benchmark-logs/docs/index_ctmc.html
[16]: https://moves-rwth.github.io/storm-benchmark-logs/docs/index_mdp.html
[17]: https://moves-rwth.github.io/storm-benchmark-logs/docs/index_ma.html
[18]: https://www.github.com/moves-rwth/storm-benchmark-logs/
[19]: https://doi.org/10.1007/978-3-030-83723-5_15
[20]: https://qcomp.org/competition/2020/
[21]: https://doi.org/10.1007/s10009-021-00633-z
[22]: https://arxiv.org/abs/2002.07080
[23]: https://doi.org/10.5281/zenodo.4017717
[24]: https://stormchecker.org/benchmarks.html
[25]: https://doi.org/10.1007/978-3-030-17502-3_5
[26]: https://doi.org/10.5281/zenodo.2628301
[27]: https://qcomp.org/competition/2019/
[28]: https://doi.org/10.1007/978-3-319-63390-9_31
[29]: https://arxiv.org/abs/1702.04311
[30]: https://stormchecker.org/benchmarks.html
