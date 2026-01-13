# The Tamarin Prover

Tamarin has been successfully used to analyze and support the development of modern security
protocols [[PDF][1], [PDF][2]], including TLS 1.3 [[PDF][3], [PDF][4]], 5G‑AKA [[PDF][5], [PDF][6]],
Noise [[PDF][7]], EMV (Chip-and-pin) [[PDF][8]], and Apple iMessage [[Apple blog][9], [PDF][10]].

### Extensive graphical user interface

Automatically find attacks or proofs. Attack graphs show exactly how a property can be violated.

### Interactive proof construction or attack finding

Construct partial proofs, or guide proof/attack search manually. For complex protocols, users can
inspect partial proofs and write auxiliary lemmas.

### Command-line interface

Perform protocol analysis in batch mode using the commandline.

### Model protocol state machines

Tamarin's core modeling mechanism uses [multiset rewrite rules][11]. Alternatively, specify
protocols using [a process calculus][12], which are automatically translated into rewrite rules.

### Specify security properties

Security properties can be specified using a [first-order logic with quantification over
timepoints][13].

Want to learn more? Consult Tamarin's extensive [documentation][14].

[1]: https://hal.science/hal-03586826v1/document
[2]: https://ethz.ch/content/dam/ethz/special-interest/infk/inst-infsec/information-security-group-d
am/research/publications/pub2017/siglog-tamarin.pdf
[3]: https://tls13tamarin.github.io/TLS13Tamarin/docs/tls13tamarin-draft21.pdf
[4]: https://cispa.saarland/group/cremers/downloads/papers/CHSV2016-TLS13.pdf
[5]: https://arxiv.org/pdf/1806.10360
[6]: https://cispa.saarland/group/cremers/downloads/papers/CrDe2018-5G.pdf
[7]: https://cispa.saarland/group/cremers/downloads/papers/Noise-Usenix2020.pdf
[8]: https://arxiv.org/pdf/2006.08249.pdf
[9]: https://security.apple.com/blog/imessage-pq3/
[10]: https://www.usenix.org/system/files/usenixsecurity25-linker.pdf
[11]: https://tamarin-prover.com/manual/master/book/005_protocol-specification-rules.html
[12]: https://tamarin-prover.com/manual/master/book/006_protocol-specification-processes.html
[13]: https://tamarin-prover.com/manual/master/book/007_property-specification.html
[14]: /documentation.html
